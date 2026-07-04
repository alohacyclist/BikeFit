"""FF3 — Inferenzlatenz: Warmup-Filter, einmaliger MAD-Ausreißerfilter,
Nearest-Rank-Perzentile (manuell), Per-Lauf-Kennzahlen + Aggregation über Läufe.

Konventionen (bewusst, dokumentiert):
  * popSD = Populations-Standardabweichung (÷N, np.std ddof=0).
  * p95 = Nearest-Rank: sorted[ceil(0.95·n) − 1] (KEINE Interpolation).
  * FPS = 1000 / mean_ms (Durchsatz-Äquivalent, nicht Mittel der per-Frame-fps).
  * meets_30fps = fps >= 30 (⇔ mean_ms <= 33,33 ms).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

FPS_THRESHOLD = 30.0
MAD_SCALE = 1.4826  # Konsistenz mit σ bei Normalverteilung.


def nearest_rank_p95(values) -> float:
    """Nearest-Rank-P95: sorted[ceil(0.95·n) − 1], geklammert auf [0, n−1]."""
    arr = np.sort(np.asarray(values, dtype=float))
    n = arr.size
    if n == 0:
        return 0.0
    idx = math.ceil(0.95 * n) - 1
    idx = min(max(idx, 0), n - 1)
    return float(arr[idx])


def mad_filter(values, k: float = 3.0):
    """Einmaliger (nicht-iterativer) MAD-Filter auf der Latenz.

    Verwirft t_i mit |t_i − median| > k · 1.4826 · MAD. Bei MAD == 0 (keine
    messbare Streuung) wird NICHT gefiltert. Returns (kept, outlier_mask).
    """
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return x, np.zeros(0, dtype=bool)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0.0:
        return x, np.zeros(x.size, dtype=bool)
    threshold = k * MAD_SCALE * mad
    outlier = np.abs(x - med) > threshold
    return x[~outlier], outlier


def latency_stats_for_run(frames_df: pd.DataFrame, k: float = 3.0) -> dict:
    """Latenz-Kennzahlen einer Session aus den Non-Warmup-Frames (nach MAD)."""
    non_warmup = frames_df[~frames_df["isWarmup"]]
    lat = non_warmup["inferenceMs"].to_numpy(dtype=float)
    n_warmup = int(frames_df["isWarmup"].sum())

    kept, outlier = mad_filter(lat, k=k)
    stats = {
        "n_frames": int(lat.size),  # Non-Warmup-Basis vor MAD
        "n_warmup": n_warmup,
        "n_outlier": int(outlier.sum()),
        "mean_ms": 0.0,
        "sd_ms": 0.0,
        "median_ms": 0.0,
        "mad_ms": 0.0,
        "p95_ms": 0.0,
        "fps": 0.0,
        "meets_30fps": False,
    }
    if kept.size == 0:
        return stats

    mean = float(np.mean(kept))
    med = float(np.median(kept))
    stats.update(
        mean_ms=mean,
        sd_ms=float(np.std(kept, ddof=0)),
        median_ms=med,
        mad_ms=float(np.median(np.abs(kept - med))),
        p95_ms=nearest_rank_p95(kept),
        fps=(1000.0 / mean) if mean > 0 else 0.0,
    )
    stats["meets_30fps"] = bool(stats["fps"] >= FPS_THRESHOLD)
    return stats


def aggregate_latency(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregation über Läufe je (level, threading).

    mean-of-means + Sample-SD-of-means (ddof=1; 0.0 bei <2 Läufen),
    median-of-medians, p95-of-p95 (Nearest-Rank über die Per-Lauf-P95).
    """
    rows = []
    for (level, threading), grp in summary_df.groupby(
        ["level", "threading"], sort=False
    ):
        means = grp["mean_ms"].to_numpy(dtype=float)
        rows.append(
            {
                "level": level,
                "threading": threading,
                "n_runs": int(len(grp)),
                "mean_of_means_ms": float(np.mean(means)),
                "sd_of_means_ms": (
                    float(np.std(means, ddof=1)) if means.size > 1 else 0.0
                ),
                "median_of_medians_ms": float(np.median(grp["median_ms"])),
                "p95_of_p95_ms": nearest_rank_p95(grp["p95_ms"].to_numpy()),
            }
        )
    return pd.DataFrame(rows)
