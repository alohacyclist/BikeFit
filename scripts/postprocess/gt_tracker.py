"""Parser für Tracker-Multi-Point-Export (Physlets Tracker) als Ground Truth.

Format:
  Zeile 1: '#multi:'
  Zeile 2: '.Masse A....Masse B....Masse C....'
  Zeile 3: '.t.x.y.θ.frame.x.y.θ.frame.x.y.θ.frame.'
  Datenzeilen: 13 Felder, Delimiter '.', Dezimaltrenner ',', Exponent 'E'.
    Beispiel-Feld '4,571334E2' == 457.1334.
  Reihenfolge je Zeile: t, x_A,y_A,θ_A,frame_A, x_B,y_B,θ_B,frame_B,
                        x_C,y_C,θ_C,frame_C.
  A=Hüfte, B=Knie, C=Sprunggelenk (links, Sagittalebene). θ_* = Punkt-
  Orientierung → ignoriert. frame 1-basiert. Sampling: t == (frame−1)/fps.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd

from .angles import knee_angle

_HEADER_LINES = 3
_FIELDS_PER_ROW = 13
_ENCODING_FALLBACKS = ("utf-8", "windows-1252", "latin-1")


class GtParseError(ValueError):
    """Fehler beim Parsen der Ground-Truth — führt zu Exit != 0."""


def detect_encoding(path: str) -> str:
    """Encoding via chardet, mit dokumentierter Fallback-Kette."""
    try:
        import chardet
    except ImportError:
        return "utf-8"
    with open(path, "rb") as handle:
        raw = handle.read()
    guess = chardet.detect(raw)
    enc = (guess or {}).get("encoding")
    return enc if enc else "utf-8"


def _read_lines(path: str) -> Tuple[list, str]:
    """Liest die Datei robust; probiert erkannte + Fallback-Encodings."""
    candidates = [detect_encoding(path), *_ENCODING_FALLBACKS]
    seen = set()
    last_err: Exception | None = None
    for enc in candidates:
        key = (enc or "").lower()
        if not enc or key in seen:
            continue
        seen.add(key)
        try:
            with open(path, "r", encoding=enc) as handle:
                return handle.read().splitlines(), enc
        except (UnicodeDecodeError, LookupError) as err:
            last_err = err
    raise GtParseError(f"{path}: kein Encoding lesbar ({last_err}).")


def _parse_field(token: str, lineno: int) -> float:
    """Ein Tracker-Feld → float. ',' → '.', 'E'-Exponent bleibt."""
    cleaned = token.strip().replace(",", ".")
    try:
        return float(cleaned)
    except ValueError as err:
        raise GtParseError(
            f"Zeile {lineno}: Feld {token!r} ist keine Zahl."
        ) from err


def parse_tracker_multi(
    path: str, target_fps: int = 30
) -> Tuple[pd.DataFrame, str]:
    """Parst den Tracker-Export → (DataFrame, verwendetes Encoding).

    Spalten: frame_gt, t_gt, hip_x, hip_y, knee_x, knee_y, ankle_x, ankle_y,
             gt_kneeAngle.
    """
    lines, encoding = _read_lines(path)
    data_lines = lines[_HEADER_LINES:]

    records = []
    for offset, raw in enumerate(data_lines):
        lineno = _HEADER_LINES + offset + 1
        stripped = raw.strip()
        if not stripped:
            continue
        # Führende/abschließende Delimiter entfernen, dann an '.' splitten.
        # Werte enthalten nie '.', da Dezimaltrenner ',' ist.
        tokens = stripped.strip(".").split(".")
        if len(tokens) != _FIELDS_PER_ROW:
            raise GtParseError(
                f"Zeile {lineno}: {len(tokens)} Felder statt {_FIELDS_PER_ROW}."
            )
        vals = [_parse_field(tok, lineno) for tok in tokens]
        t = vals[0]
        hip = (vals[1], vals[2])
        knee = (vals[5], vals[6])
        ankle = (vals[9], vals[10])
        frame_gt = int(round(vals[4]))  # frame_A (1-basiert)
        records.append(
            {
                "frame_gt": frame_gt,
                "t_gt": t,
                "hip_x": hip[0],
                "hip_y": hip[1],
                "knee_x": knee[0],
                "knee_y": knee[1],
                "ankle_x": ankle[0],
                "ankle_y": ankle[1],
                "gt_kneeAngle": knee_angle(hip, knee, ankle),
            }
        )

    if not records:
        raise GtParseError(f"{path}: keine Datenzeilen nach Header.")

    df = pd.DataFrame(records)

    # Sanity: t == (frame-1)/fps.
    drift = (df["t_gt"] - (df["frame_gt"] - 1) / target_fps).abs().max()
    if drift >= 1e-3:
        raise GtParseError(
            f"{path}: t/frame-Inkonsistenz (max drift {drift:.4g} s bei "
            f"{target_fps} fps)."
        )

    return df, encoding


def apply_frame_offset(df: pd.DataFrame, offset: int) -> pd.DataFrame:
    """Fügt frameIndex = frame_gt + offset hinzu (Default-Offset -1)."""
    out = df.copy()
    out["frameIndex"] = out["frame_gt"] + offset
    return out
