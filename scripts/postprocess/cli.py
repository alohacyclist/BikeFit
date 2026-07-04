"""CLI-Argumente + Einstiegspunkt für das Post-Processing."""

from __future__ import annotations

import argparse
from typing import Optional

from . import report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "BikeFit Post-Processing (Vertrag v1.0.0): FF1/FF2/FF3 aus 3 "
            "JSON-Stufen + Tracker-Ground-Truth."
        )
    )
    parser.add_argument("--proband", required=True, help="Proband-ID, z.B. P01.")
    parser.add_argument("--gt", required=True, help="Pfad zur GT-Datei.")
    parser.add_argument(
        "--gt-format", default="tracker-multi", choices=["tracker-multi"],
        help="GT-Format (aktuell nur tracker-multi).",
    )
    parser.add_argument(
        "--gt-frame-offset", type=int, default=-1,
        help="frameIndex = frame_gt + offset (Default -1: GT 1- → 0-basiert).",
    )
    parser.add_argument("--fp32", default=None, help="Glob der FP32-Session-JSONs.")
    parser.add_argument("--fp16", default=None, help="Glob der FP16-Session-JSONs.")
    parser.add_argument("--int8", default=None, help="Glob der INT8-Session-JSONs.")
    parser.add_argument(
        "--side", default="left", choices=["left", "right"],
        help="Ausgewertete Körperseite (gewinnt über session.bodySide).",
    )
    parser.add_argument(
        "--warmup", type=int, default=5,
        help="Warmup-Frames (Cross-Check; isWarmup-Flag ist maßgeblich).",
    )
    parser.add_argument(
        "--outlier-k", type=float, default=3.0,
        help="MAD-Faktor k für den Latenz-Ausreißerfilter (Default 3.0).",
    )
    parser.add_argument("--out", required=True, help="Ausgabeverzeichnis.")
    parser.add_argument(
        "--strict", action="store_true",
        help="Hart abbrechen bei Frame-Zahl-Diskrepanz GT vs Pipeline.",
    )
    return parser


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return report.run(args)
