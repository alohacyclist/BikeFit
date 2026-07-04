"""Sekundär, deskriptiv — Pedalzyklen: BDC-Detektion + Zyklus-Genauigkeit.

BDC (Bottom Dead Center) = Kurbel unten → Bein maximal gestreckt → Kniewinkel
MAXIMAL (Innenwinkel nahe 180°). Detektion via scipy.signal.find_peaks auf dem
FP32-Genauigkeits-Sample (Maxima). Zyklus = BDC-zu-BDC.

scipy.signal.find_peaks wird AUSSCHLIESSLICH hier für die Peak-Detektion
verwendet — Perzentile bleiben Nearest-Rank (siehe latency.py).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .accuracy import mae_rmse

# Defaults für die BDC-Detektion — konservativ; via CLI/report übersteuerbar.
DEFAULT_PROMINENCE = 5.0  # Grad
DEFAULT_MIN_DISTANCE_FRAMES = 10  # ~0,33 s @30fps < ein Tritt bei 90 rpm (~20 F)


def detect_bdc_frames(
    joined_fp32: pd.DataFrame,
    prominence: float = DEFAULT_PROMINENCE,
    distance: int = DEFAULT_MIN_DISTANCE_FRAMES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """BDC-Frames aus dem FP32-Sample. Returns (bdc_frameIndices, angle, fidx)."""
    d = joined_fp32.sort_values("frameIndex")
    angle = d["pipeline_angle"].to_numpy(dtype=float)
    fidx = d["frameIndex"].to_numpy(dtype=int)
    if angle.size == 0:
        return np.array([], dtype=int), angle, fidx
    peaks, _ = find_peaks(
        angle, prominence=prominence, distance=max(1, distance)
    )
    return fidx[peaks], angle, fidx


def _bdc_value(jdf: pd.DataFrame, frame: int) -> Tuple[float, float]:
    """Pipeline- und GT-Winkel am BDC-Frame (erste Übereinstimmung)."""
    row = jdf[jdf["frameIndex"] == frame]
    if row.empty:
        return float("nan"), float("nan")
    return float(row["pipeline_angle"].iloc[0]), float(row["gt_kneeAngle"].iloc[0])


def cycle_metrics(
    bdc_frames: np.ndarray, joined_by_level: Dict[str, pd.DataFrame]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Zyklus- und BDC-Kennzahlen je Stufe.

    Returns (cycle_summary_df, bdc_angle_per_cycle_df).
    Zyklen = aufeinanderfolgende BDC-Frames (aus FP32 detektiert), auf ALLE
    Stufen angewandt (identische Frame-Grenzen → vergleichbar).
    """
    cycle_rows = []
    bdc_rows = []
    for level, jdf in joined_by_level.items():
        jdf = jdf.sort_values("frameIndex")

        for ci, bf in enumerate(bdc_frames):
            pip, gt = _bdc_value(jdf, int(bf))
            bdc_rows.append(
                {
                    "cycle_idx": ci,
                    "level": level,
                    "bdc_frame": int(bf),
                    "bdc_pipeline_deg": pip,
                    "bdc_gt_deg": gt,
                    "delta_deg": (pip - gt) if np.isfinite(pip) and np.isfinite(gt)
                    else float("nan"),
                }
            )

        for ci in range(len(bdc_frames) - 1):
            start, end = int(bdc_frames[ci]), int(bdc_frames[ci + 1])
            seg = jdf[(jdf["frameIndex"] >= start) & (jdf["frameIndex"] <= end)]
            if seg.empty:
                continue
            mae, rmse = mae_rmse(seg["diff"])
            pip = seg["pipeline_angle"].to_numpy(dtype=float)
            cycle_rows.append(
                {
                    "cycle_idx": ci,
                    "start_frame": start,
                    "end_frame": end,
                    "level": level,
                    "MAE_deg": mae,
                    "RMSE_deg": rmse,
                    "amplitude_deg": float(np.max(pip) - np.min(pip)),
                }
            )

    return pd.DataFrame(cycle_rows), pd.DataFrame(bdc_rows)


def aggregate_cycles(cycle_summary: pd.DataFrame) -> pd.DataFrame:
    """Über Zyklen je Stufe: MAE/RMSE mean + SD (Stichproben-SD, ddof=1)."""
    if cycle_summary.empty:
        return pd.DataFrame(
            columns=[
                "level", "n_cycles", "MAE_cycle_mean", "MAE_cycle_SD",
                "RMSE_cycle_mean", "RMSE_cycle_SD",
            ]
        )
    rows = []
    for level, grp in cycle_summary.groupby("level", sort=False):
        mae = grp["MAE_deg"].to_numpy(dtype=float)
        rmse = grp["RMSE_deg"].to_numpy(dtype=float)
        rows.append(
            {
                "level": level,
                "n_cycles": int(len(grp)),
                "MAE_cycle_mean": float(np.mean(mae)),
                "MAE_cycle_SD": float(np.std(mae, ddof=1)) if mae.size > 1 else 0.0,
                "RMSE_cycle_mean": float(np.mean(rmse)),
                "RMSE_cycle_SD": float(np.std(rmse, ddof=1)) if rmse.size > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)
