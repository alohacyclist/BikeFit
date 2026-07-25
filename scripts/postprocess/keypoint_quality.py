"""Sekundär-Analyse Keypoint-Qualität (deskriptiv, rein additiv).

Quantifiziert die Detection-Zuverlässigkeit der winkelrelevanten Keypoints
(hip/knee/ankle der gewählten Seite). KEIN Filter auf die Haupt-MAE/RMSE — alle
Kennzahlen hier sind zusätzlich (R14: keine Selektionseffekte). Die
score-gewichtete MAE ersetzt NICHT die frame-weise MAE, sie ergänzt sie nur.

Konvention: Scores über ALLE Frames (inkl. Warmup) — Warmup betrifft nur das
Latenz-Timing, nicht die Detection. Plots markieren Warmup separat grau.
SD = Populations-SD (ddof=0), konsistent zur Latenz-Auswertung.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .angles import SIDE_KEYPOINTS

DEFAULT_SCORE_THRESHOLD = 0.5
WORST_CASE_ANKLE_MAX = 0.3  # Schwelle für Worst-Case-Frame-Auswahl (fix).

ROLES = ("hip", "knee", "ankle")

# Kurznamen der winkelrelevanten Keypoints je Seite (CSV/Plot-Labels).
SIDE_KEYPOINT_NAMES = {
    "left": {"hip": "lhip", "knee": "lknee", "ankle": "lankle"},
    "right": {"hip": "rhip", "knee": "rknee", "ankle": "rankle"},
}

KEYPOINT_QUALITY_COLUMNS = [
    "level", "threading", "runIndex", "keypoint_idx", "keypoint_name",
    "mean_score", "median_score", "sd_score",
    "n_frames_low_conf", "pct_frames_low_conf",
]


def side_indices(side: str) -> Dict[str, int]:
    """role→COCO-Index für die gewählte Seite (Kopie, keine Mutation)."""
    return dict(SIDE_KEYPOINTS[side])


def keypoint_quality_rows(
    level: str,
    threading: str,
    run_index: int,
    scores_df: pd.DataFrame,
    side: str,
    threshold: float,
) -> List[dict]:
    """Eine Zeile je winkelrelevantem Keypoint (über ALLE Frames der Session)."""
    idxs = side_indices(side)
    names = SIDE_KEYPOINT_NAMES[side]
    n_total = int(len(scores_df))
    rows: List[dict] = []
    for role in ROLES:
        s = scores_df[f"score_{role}"].to_numpy(dtype=float)
        low = int(np.sum(s < threshold))
        rows.append(
            {
                "level": level,
                "threading": threading,
                "runIndex": run_index,
                "keypoint_idx": idxs[role],
                "keypoint_name": names[role],
                "mean_score": float(np.mean(s)) if s.size else float("nan"),
                "median_score": float(np.median(s)) if s.size else float("nan"),
                "sd_score": float(np.std(s, ddof=0)) if s.size else float("nan"),
                "n_frames_low_conf": low,
                "pct_frames_low_conf": (100.0 * low / n_total)
                if n_total else float("nan"),
            }
        )
    return rows


def session_score_summary(
    scores_df: pd.DataFrame, side: str, threshold: float
) -> dict:
    """Zusatzspalten für summary.csv: mean-Score je Keypoint + %<Schwelle Ankle."""
    names = SIDE_KEYPOINT_NAMES[side]
    out: dict = {}
    for role in ROLES:
        s = scores_df[f"score_{role}"].to_numpy(dtype=float)
        out[f"{names[role]}_mean_score"] = (
            float(np.mean(s)) if s.size else float("nan")
        )
    ankle = scores_df["score_ankle"].to_numpy(dtype=float)
    n = ankle.size
    low = int(np.sum(ankle < threshold))
    out[f"pct_{names['ankle']}_below_0_5"] = (
        (100.0 * low / n) if n else float("nan")
    )
    return out


def summary_score_columns(side: str) -> List[str]:
    """Spaltennamen (Reihenfolge) der summary.csv-Score-Ergänzung je Seite."""
    names = SIDE_KEYPOINT_NAMES[side]
    return [f"{names[r]}_mean_score" for r in ROLES] + [
        f"pct_{names['ankle']}_below_0_5"
    ]


def weighted_mae(abs_diff, weights) -> float:
    """Σ(w·|Δ|)/Σw. w = Produkt der Keypoint-Scores. nan bei Σw<=0 / leer.

    Deskriptive Sekundär-Kennzahl — KEIN Ersatz der frame-weisen MAE.
    """
    a = np.asarray(abs_diff, dtype=float)
    w = np.asarray(weights, dtype=float)
    denom = float(np.sum(w))
    if a.size == 0 or not np.isfinite(denom) or denom <= 0.0:
        return float("nan")
    return float(np.sum(w * a) / denom)


def pearson_r(x, y) -> float:
    """Pearson-Korrelation (deskriptiv, numpy). nan bei n<2 oder Nullvarianz.

    Kein p-value: rein deskriptives Assoziationsmaß (Frame-Autokorrelation
    machte einen p-Wert ohnehin methodisch fragwürdig).
    """
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def worst_case_frame(
    df: pd.DataFrame, ankle_max: float = WORST_CASE_ANKLE_MAX, side: str = "left"
) -> Optional[dict]:
    """Frame mit maximalem |Δθ| UND score_ankle < ankle_max (0.3). None wenn keiner.

    df braucht: frameIndex, abs_delta, score_ankle, threading, runIndex.
    Der Ankle-Score-Key ist seitenabhängig (lankle_score / rankle_score).
    """
    cand = df[df["score_ankle"] < ankle_max]
    if cand.empty:
        return None
    row = cand.loc[cand["abs_delta"].idxmax()]
    ankle_key = f"{SIDE_KEYPOINT_NAMES[side]['ankle']}_score"
    return {
        "frameIndex": int(row["frameIndex"]),
        "delta_deg": float(row["abs_delta"]),
        ankle_key: float(row["score_ankle"]),
        "threading": str(row["threading"]),
        "runIndex": int(row["runIndex"]),
    }
