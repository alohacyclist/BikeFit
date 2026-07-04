#!/usr/bin/env python3
"""BikeFit Post-Processing — Einstiegspunkt (Vertrag v1.0.0).

Konsumiert die drei JSON-Stufen (fp32/fp16/int8) einer Voll-Sequenz plus die
Tracker-Ground-Truth und berechnet FF1/FF2/FF3. Die Logik liegt im Package
`postprocess/` (viele kleine Module); diese Datei ist nur der CLI-Wrapper.

Beispiel:
    python3 scripts/postprocess_benchmark.py \\
        --proband P01 \\
        --gt data/gt/P01_tracker.txt \\
        --gt-format tracker-multi \\
        --gt-frame-offset -1 \\
        --fp32 "exports/benchmark_P01_fp32_*.json" \\
        --fp16 "exports/benchmark_P01_fp16_*.json" \\
        --int8 "exports/benchmark_P01_int8_*.json" \\
        --side left --warmup 5 --outlier-k 3.0 \\
        --out results/P01/
"""

from __future__ import annotations

import os
import sys

# Package-Verzeichnis (scripts/) auf den Pfad legen, egal von wo aufgerufen.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from postprocess.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
