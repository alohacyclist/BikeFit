"""Laden + Fail-fast-Validierung der Benchmark-Session-JSONs (Vertrag v1.0.0).

session.level ist die Single Source of Truth; der Dateiname wird nur zur
Konsistenz gegengeprüft (Warnung bei Abweichung, kein Abbruch).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

import pandas as pd

SCHEMA_VERSION = "1.0.0"
LEVELS = ("fp32", "fp16", "int8")
SIDES = ("left", "right")
THREADINGS = ("single", "multi")
DROP_REASONS = ("seek_timeout", "predict_error", "readback_error", "other")
KEYPOINT_COUNT = 17

_REQUIRED_SESSION_KEYS = (
    "schemaVersion",
    "probandId",
    "level",
    "runIndex",
    "createdAt",
    "targetFps",
    "bodySide",
    "threading",
    "videoDurationSec",
    "videoTotalFrames",
    "warmupCount",
    "modelFingerprintSha256",
    "modelUrl",
    "modelLoadMs",
    "userAgent",
    "hardware",
    "frames",
    "droppedFrames",
)

_REQUIRED_FRAME_KEYS = (
    "frameIndex",
    "timestampMs",
    "inferenceMs",
    "fps",
    "kneeAngleRight",
    "kneeAngleLeft",
    "keypoints",
    "keypointScores",
    "isWarmup",
)

# Dateiname: benchmark_<probandId>_<level>_<stamp>.json
_FILENAME_RE = re.compile(
    r"^benchmark_(?P<proband>[^_]+)_(?P<level>fp32|fp16|int8)_(?P<stamp>.+)\.json$"
)


def _is_number(value: Any) -> bool:
    """True für echte Zahlen — bool ist in Python int, hier aber keine Zahl."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class SchemaError(ValueError):
    """Verstoß gegen den Vertrag v1.0.0 — führt zu Exit != 0."""


def parse_level_from_filename(path: str) -> Optional[str]:
    """Extrahiert die Stufe aus dem Dateinamen (nur Konsistenzcheck)."""
    match = _FILENAME_RE.match(os.path.basename(path))
    return match.group("level") if match else None


def _validate_frame(frame: Any, i: int, source: str) -> None:
    if not isinstance(frame, dict):
        raise SchemaError(f"{source}: frames[{i}] ist kein Objekt.")
    for key in _REQUIRED_FRAME_KEYS:
        if key not in frame:
            raise SchemaError(f"{source}: frames[{i}] fehlt '{key}'.")
    for num_key in ("frameIndex", "timestampMs", "inferenceMs", "fps",
                    "kneeAngleRight", "kneeAngleLeft"):
        if not _is_number(frame[num_key]):
            raise SchemaError(
                f"{source}: frames[{i}].{num_key} ist keine Zahl."
            )
    if not isinstance(frame["isWarmup"], bool):
        raise SchemaError(f"{source}: frames[{i}].isWarmup ist kein Boolean.")
    kps = frame["keypoints"]
    if not isinstance(kps, list) or len(kps) != KEYPOINT_COUNT:
        raise SchemaError(
            f"{source}: frames[{i}].keypoints muss Länge {KEYPOINT_COUNT} haben."
        )
    for j, kp in enumerate(kps):
        if not isinstance(kp, dict) or not _is_number(kp.get("x")) \
                or not _is_number(kp.get("y")):
            raise SchemaError(
                f"{source}: frames[{i}].keypoints[{j}] braucht numerische x,y."
            )
    scores = frame["keypointScores"]
    if not isinstance(scores, list) or len(scores) != KEYPOINT_COUNT:
        raise SchemaError(
            f"{source}: frames[{i}].keypointScores muss Länge {KEYPOINT_COUNT} haben."
        )


def validate_session(session: Any, source: str = "<session>") -> None:
    """Fail-fast-Validierung. Wirft SchemaError bei Vertragsverstoß."""
    if not isinstance(session, dict):
        raise SchemaError(f"{source}: Session ist kein JSON-Objekt.")

    for key in _REQUIRED_SESSION_KEYS:
        if key not in session:
            raise SchemaError(f"{source}: Pflichtfeld '{key}' fehlt.")

    if session["schemaVersion"] != SCHEMA_VERSION:
        raise SchemaError(
            f"{source}: schemaVersion {session['schemaVersion']!r} "
            f"!= {SCHEMA_VERSION!r}."
        )
    if session["level"] not in LEVELS:
        raise SchemaError(f"{source}: level {session['level']!r} ungültig.")
    if session["bodySide"] not in SIDES:
        raise SchemaError(f"{source}: bodySide {session['bodySide']!r} ungültig.")
    if session["threading"] not in THREADINGS:
        raise SchemaError(
            f"{source}: threading {session['threading']!r} ungültig."
        )
    if not isinstance(session["runIndex"], int) or session["runIndex"] < 0:
        raise SchemaError(f"{source}: runIndex muss int >= 0 sein.")

    hardware = session["hardware"]
    if not isinstance(hardware, dict):
        raise SchemaError(f"{source}: hardware ist kein Objekt.")
    for hk in ("device", "cpu", "os", "browser"):
        if hk not in hardware:
            raise SchemaError(f"{source}: hardware.{hk} fehlt.")

    frames = session["frames"]
    if not isinstance(frames, list):
        raise SchemaError(f"{source}: 'frames' ist keine Liste.")
    if not frames:
        raise SchemaError(f"{source}: 'frames' ist leer — keine Messdaten.")

    dropped = session["droppedFrames"]
    if not isinstance(dropped, list):
        raise SchemaError(f"{source}: 'droppedFrames' ist keine Liste.")
    for k, d in enumerate(dropped):
        if not isinstance(d, dict) or "frameIndex" not in d or "reason" not in d:
            raise SchemaError(
                f"{source}: droppedFrames[{k}] braucht frameIndex + reason."
            )
        if d["reason"] not in DROP_REASONS:
            raise SchemaError(
                f"{source}: droppedFrames[{k}].reason {d['reason']!r} ungültig."
            )

    for i, frame in enumerate(frames):
        _validate_frame(frame, i, source)

    # Invariante: videoTotalFrames == frames + droppedFrames.
    expected = len(frames) + len(dropped)
    if session["videoTotalFrames"] != expected:
        raise SchemaError(
            f"{source}: Invariante verletzt — videoTotalFrames="
            f"{session['videoTotalFrames']} != frames({len(frames)}) + "
            f"dropped({len(dropped)}) = {expected}."
        )


def load_session(path: str) -> dict:
    """Liest + validiert eine Session-JSON. Hängt _source an."""
    with open(path, "r", encoding="utf-8") as handle:
        session = json.load(handle)
    validate_session(session, source=path)
    session["_source"] = path
    session["_filenameLevel"] = parse_level_from_filename(path)
    return session


def frames_dataframe(session: dict) -> pd.DataFrame:
    """ALLE Frames als DataFrame (inkl. Warmup — Filterung erfolgt downstream)."""
    rows = [
        {
            "frameIndex": int(f["frameIndex"]),
            "timestampMs": float(f["timestampMs"]),
            "inferenceMs": float(f["inferenceMs"]),
            "fps": float(f["fps"]),
            "kneeAngleLeft": float(f["kneeAngleLeft"]),
            "kneeAngleRight": float(f["kneeAngleRight"]),
            "isWarmup": bool(f["isWarmup"]),
        }
        for f in session["frames"]
    ]
    return pd.DataFrame(rows)
