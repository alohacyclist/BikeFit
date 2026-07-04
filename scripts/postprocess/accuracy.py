"""FF1/FF2 — Genauigkeit: Join Pipeline↔GT, MAE/RMSE, paarweise Δθ,
Bland-Altman (rein deskriptiv). Genauigkeits-Join nutzt NUR Non-Warmup-Frames,
KEINEN Latenz-Ausreißerfilter (getrennte Stichproben, Design-Vorgabe).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from .angles import SIDE_ANGLE_COLUMN

RMSE_THRESHOLD = 10.0  # FF1-Akzeptanz: RMSE < 10°.


def join_session(
    frames_df: pd.DataFrame, gt_df: pd.DataFrame, side: str
) -> pd.DataFrame:
    """Innerer Join (frameIndex) Pipeline-Non-Warmup ↔ GT. Winkelspalte via side.

    Ergebnis-Spalten: frameIndex, pipeline_angle, gt_kneeAngle, diff.
    """
    angle_col = SIDE_ANGLE_COLUMN[side]
    non_warmup = frames_df[~frames_df["isWarmup"]][
        ["frameIndex", angle_col]
    ].rename(columns={angle_col: "pipeline_angle"})
    merged = non_warmup.merge(
        gt_df[["frameIndex", "gt_kneeAngle"]], on="frameIndex", how="inner"
    )
    merged["diff"] = merged["pipeline_angle"] - merged["gt_kneeAngle"]
    return merged


def mae_rmse(diff) -> Tuple[float, float]:
    """MAE = mean(|Δ|), RMSE = sqrt(mean(Δ²)). nan bei leerem Sample."""
    d = np.asarray(diff, dtype=float)
    if d.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(np.abs(d))), float(np.sqrt(np.mean(d ** 2)))


def bland_altman(pipeline, gt) -> dict:
    """Bland-Altman (Pipeline vs GT): Mean-Difference + LoA = mean ± 1.96·SD.

    SD = Stichproben-SD (ddof=1). Rein deskriptiv, keine Inferenzstatistik.
    """
    p = np.asarray(pipeline, dtype=float)
    g = np.asarray(gt, dtype=float)
    diff = p - g
    mean_pair = (p + g) / 2.0
    md = float(np.mean(diff)) if diff.size else float("nan")
    sd = float(np.std(diff, ddof=1)) if diff.size > 1 else 0.0
    return {
        "n": int(diff.size),
        "mean_diff": md,
        "sd_diff": sd,
        "loa_lower": md - 1.96 * sd,
        "loa_upper": md + 1.96 * sd,
        "mean_pair": mean_pair,
        "diff": diff,
    }


def pairwise_delta(
    pangles_df: pd.DataFrame, other_level: str, base_level: str = "fp32"
) -> np.ndarray:
    """Frame-weise Δθ = angle(other) − angle(base), gematcht auf
    (threading, runIndex, frameIndex). pangles_df: long-Format mit Spalten
    level, threading, runIndex, frameIndex, pipeline_angle.
    """
    keys = ["threading", "runIndex", "frameIndex"]
    base = pangles_df[pangles_df["level"] == base_level][keys + ["pipeline_angle"]]
    other = pangles_df[pangles_df["level"] == other_level][
        keys + ["pipeline_angle"]
    ]
    merged = base.merge(other, on=keys, suffixes=("_base", "_other"))
    return (
        merged["pipeline_angle_other"] - merged["pipeline_angle_base"]
    ).to_numpy(dtype=float)


def delta_stats(delta) -> dict:
    """Deskriptive Kennzahlen einer Δθ-Verteilung."""
    d = np.asarray(delta, dtype=float)
    if d.size == 0:
        return {"n": 0, "mean": float("nan"), "sd": float("nan"),
                "mae": float("nan"), "rmse": float("nan")}
    mae, rmse = mae_rmse(d)
    return {
        "n": int(d.size),
        "mean": float(np.mean(d)),
        "sd": float(np.std(d, ddof=1)) if d.size > 1 else 0.0,
        "mae": mae,
        "rmse": rmse,
    }
