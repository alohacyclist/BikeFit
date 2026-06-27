#!/usr/bin/env python3
"""Post-Processing für BikeFit-Benchmark-Exporte.

Verarbeitet die Roh-JSON-Dateien, die der BenchmarkExporter
(`src/services/BenchmarkExporter.ts`, `downloadJSON`) schreibt, und berechnet
die Latenz-/Validierungs-Statistik aus den Per-Frame-Rohdaten.

ROHFORMAT (eine Session = eine Quantisierungsstufe, ein Video, ein Durchlauf):

    {
      "participantId": "P01",
      "quantizationLevel": "fp32" | "fp16" | "int8",
      "startTimestamp": "2026-06-21T...Z",
      "systemInfo": { "userAgent": str, "hardwareConcurrency": int, ... },
      "warmupFrames": 5,                # Schwellwert: erste N Inferenzen = Warmup
      "lockedSide": "left" | "right" | null,
      "modelFingerprint": { "sha256": str, "sizeBytes": int } | null,
      "threadingMode": "single" | "multi" | "unknown",
      "numThreads": int | null,
      "videoSource": "file",
      "videoSourceName": str | undefined,
      "targetFps": 30,                  # CFR-Stepping-Rate
      "durationSeconds": float,         # Videolänge
      "expectedFrames": int,            # floor(durationSeconds * targetFps)
      "frames": [
        {
          "frameIndex": int,            # 1..N pro Aufnahme
          "timestampMs": float,         # performance.now() bei Aufzeichnung
          "inferenceMs": float,         # NUR Modell-Forward-Pass + GPU/WASM-Sync
          "fps": float,                 # per-Frame gleitender Wall-Clock-Durchsatz
          "kneeAngleRight": float,      # immer berechnet (kein App-Filter)
          "kneeAngleLeft": float,       # immer berechnet (kein App-Filter)
          "keypointScores": [float x 17],
          "isWarmup": bool              # true => aus Statistik ausschließen
        },
        ...
      ]
    }

Confidence-Filterung (Selektion gültiger Frames) erfolgt AUSSCHLIESSLICH hier
im Postprocessing — anhand keypointScores und einer dokumentierten Schwelle
(Default 0.2, per CLI variierbar für Sensitivity-Sweep). Damit kann auf
identischen Roh-Frames mit verschiedenen Schwellen reproduziert werden.

Statistik wird ausschließlich über NON-WARMUP-Frames (isWarmup == false) gebildet
— identisch zur (entfernten) In-App-Methode getSummary, mit EINER Anpassung:

    fps = 1000 / mean_inference_ms   (Latenz-Durchsatz-Äquivalent)
    statt mean(per-Frame-fps).

Konventionen (bewusst identisch zum früheren TS-getSummary):
  * stdInferenceMs = POPULATIONS-Standardabweichung (÷ N).
  * p50/p95 = nearest-rank, idx = floor(p/100 * N), auf aufsteigend sortiert,
    keine lineare Interpolation.
Beides ist hier zentral dokumentiert und leicht austauschbar (statistics.stdev
für Stichproben-SD bzw. eine Interpolations-Perzentile), falls die Thesis das
anders verlangt.

Aufruf:
    python3 scripts/postprocess_benchmark.py run_fp32.json run_fp16.json --json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from typing import Any, Optional

# Erwartete Keypoint-Anzahl (MoveNet) — siehe src/types/quantization.ts.
KEYPOINT_COUNT = 17

# Pflichtfelder pro Frame, die die Statistik benötigt.
_REQUIRED_FRAME_KEYS = ("inferenceMs", "isWarmup")


def _is_number(value: Any) -> bool:
    """True für echte Zahlen — bool ist in Python int, hier aber keine Zahl."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def load_session(path: str) -> dict:
    """Liest eine Benchmark-Session-JSON und validiert ihr Rohformat."""
    with open(path, "r", encoding="utf-8") as handle:
        session = json.load(handle)
    validate_session(session, source=path)
    return session


def validate_session(session: Any, source: str = "<session>") -> None:
    """Fail-fast-Validierung des Rohformats. Wirft ValueError bei Verstoß."""
    if not isinstance(session, dict):
        raise ValueError(f"{source}: Session ist kein JSON-Objekt.")

    frames = session.get("frames")
    if not isinstance(frames, list):
        raise ValueError(f"{source}: 'frames' fehlt oder ist keine Liste.")
    if not frames:
        raise ValueError(f"{source}: 'frames' ist leer — keine Messdaten.")

    for i, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"{source}: frames[{i}] ist kein Objekt.")
        for key in _REQUIRED_FRAME_KEYS:
            if key not in frame:
                raise ValueError(f"{source}: frames[{i}] fehlt '{key}'.")
        if not _is_number(frame["inferenceMs"]):
            raise ValueError(
                f"{source}: frames[{i}].inferenceMs ist keine Zahl."
            )
        if not isinstance(frame["isWarmup"], bool):
            raise ValueError(
                f"{source}: frames[{i}].isWarmup ist kein Boolean."
            )
        for side in ("kneeAngleRight", "kneeAngleLeft"):
            value = frame.get(side, None)
            if value is not None and not _is_number(value):
                raise ValueError(
                    f"{source}: frames[{i}].{side} ist weder Zahl noch null."
                )
        scores = frame.get("keypointScores")
        if scores is not None and not isinstance(scores, list):
            raise ValueError(
                f"{source}: frames[{i}].keypointScores ist keine Liste."
            )


def valid_frames(session: dict) -> list[dict]:
    """Non-Warmup-Frames — die einzige Basis aller Aggregate."""
    return [f for f in session["frames"] if not f["isWarmup"]]


def nearest_rank(sorted_values: list[float], p: float) -> float:
    """Nearest-rank-Perzentil (identisch zum früheren TS-getSummary).

    idx = clamp(floor(p/100 * N), 0, N-1) auf aufsteigend sortierten Werten.
    """
    n = len(sorted_values)
    if n == 0:
        return 0.0
    idx = min(n - 1, max(0, math.floor((p / 100.0) * n)))
    return sorted_values[idx]


def mean_or_none(values: list[Optional[float]]) -> Optional[float]:
    """Mittel über die Nicht-null-Werte; None falls keine vorhanden."""
    present = [v for v in values if v is not None]
    if not present:
        return None
    return statistics.fmean(present)


def summarize(session: dict) -> dict:
    """Berechnet die Kennzahlen einer Session aus den Non-Warmup-Frames."""
    all_frames = session["frames"]
    valid = valid_frames(session)

    summary: dict[str, Any] = {
        "participantId": session.get("participantId"),
        "quantizationLevel": session.get("quantizationLevel"),
        "threadingMode": session.get("threadingMode"),
        "numThreads": session.get("numThreads"),
        "targetFps": session.get("targetFps"),
        "durationSeconds": session.get("durationSeconds"),
        "expectedFrames": session.get("expectedFrames"),
        "warmupFrames": session.get("warmupFrames"),
        "totalFrames": len(all_frames),
        "validFrames": len(valid),
        "meanInferenceMs": 0.0,
        "stdInferenceMs": 0.0,
        "p50Ms": 0.0,
        "p95Ms": 0.0,
        "fps": 0.0,
        "meanKneeAngleRight": None,
        "meanKneeAngleLeft": None,
    }

    if not valid:
        return summary

    latencies = [float(f["inferenceMs"]) for f in valid]
    sorted_latencies = sorted(latencies)
    mean_latency = statistics.fmean(latencies)

    summary["meanInferenceMs"] = mean_latency
    # Populations-SD (÷N) — wie früheres getSummary. pstdev braucht >=1 Wert.
    summary["stdInferenceMs"] = statistics.pstdev(latencies) if len(latencies) > 1 else 0.0
    summary["p50Ms"] = nearest_rank(sorted_latencies, 50)
    summary["p95Ms"] = nearest_rank(sorted_latencies, 95)
    # ANPASSUNG: Durchsatz aus mittlerer Latenz, NICHT Mittel der per-Frame-fps.
    summary["fps"] = 1000.0 / mean_latency if mean_latency > 0 else 0.0
    summary["meanKneeAngleRight"] = mean_or_none(
        [f.get("kneeAngleRight") for f in valid]
    )
    summary["meanKneeAngleLeft"] = mean_or_none(
        [f.get("kneeAngleLeft") for f in valid]
    )
    return summary


def _fmt(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "–"
    return f"{value:.{digits}f}"


def format_summary(summary: dict) -> str:
    """Menschlich lesbarer Block für eine Session."""
    lines = [
        f"Teilnehmer:    {summary['participantId']}",
        f"Stufe:         {summary['quantizationLevel']}"
        f"  (threading={summary['threadingMode']}, numThreads={summary['numThreads']})",
        f"Video:         {_fmt(summary['durationSeconds'], 1)} s @ "
        f"{summary['targetFps']} fps  (erwartet {summary['expectedFrames']} Frames)",
        f"Frames:        {summary['totalFrames']} total, "
        f"{summary['validFrames']} valide (Warmup-Schwelle {summary['warmupFrames']})",
        f"Latenz:        mean {_fmt(summary['meanInferenceMs'])} ms  "
        f"± {_fmt(summary['stdInferenceMs'])} (pop-SD)",
        f"               p50 {_fmt(summary['p50Ms'])} ms  "
        f"p95 {_fmt(summary['p95Ms'])} ms",
        f"Durchsatz:     {_fmt(summary['fps'], 1)} fps  (= 1000 / mean_latency)",
        f"Kniewinkel ⌀:  rechts {_fmt(summary['meanKneeAngleRight'], 1)}°  "
        f"links {_fmt(summary['meanKneeAngleLeft'], 1)}°",
    ]
    return "\n".join(lines)


def format_comparison(summaries: list[dict]) -> str:
    """Kompakte Vergleichstabelle über mehrere Stufen/Sessions."""
    header = (
        f"{'Stufe':<8}{'mean ms':>10}{'pop-SD':>9}"
        f"{'p50':>8}{'p95':>8}{'fps':>9}{'valide':>9}"
    )
    rows = [header, "-" * len(header)]
    for s in summaries:
        rows.append(
            f"{str(s['quantizationLevel']):<8}"
            f"{_fmt(s['meanInferenceMs']):>10}"
            f"{_fmt(s['stdInferenceMs']):>9}"
            f"{_fmt(s['p50Ms']):>8}"
            f"{_fmt(s['p95Ms']):>8}"
            f"{_fmt(s['fps'], 1):>9}"
            f"{s['validFrames']:>9}"
        )
    return "\n".join(rows)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregiert BikeFit-Benchmark-Rohexporte (Latenz/Validierung)."
    )
    parser.add_argument("paths", nargs="+", help="Eine oder mehrere Session-JSONs.")
    parser.add_argument(
        "--json", action="store_true", help="Ergebnis als JSON statt Text ausgeben."
    )
    args = parser.parse_args(argv)

    summaries: list[dict] = []
    for path in args.paths:
        try:
            session = load_session(path)
        except (OSError, ValueError, json.JSONDecodeError) as err:
            print(f"FEHLER ({path}): {err}", file=sys.stderr)
            return 1
        summaries.append(summarize(session))

    if args.json:
        print(json.dumps(summaries, indent=2, ensure_ascii=False))
        return 0

    for path, summary in zip(args.paths, summaries):
        print(f"=== {path} ===")
        print(format_summary(summary))
        print()
    if len(summaries) > 1:
        print(format_comparison(summaries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
