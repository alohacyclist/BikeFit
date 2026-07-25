"""Matplotlib-Abbildungen (Agg-Backend, headless). Jede Funktion schreibt EIN
PNG. Robust gegen leere Samples (überspringt dann still mit Rückgabe False).
"""

from __future__ import annotations

from typing import Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from .latency import mad_filter  # noqa: E402

# Farbe je Keypoint-Rolle (Score-Plots) und je Stufe (delta_vs_score).
_ROLE_COLORS = {"hip": "#2266cc", "knee": "#16a34a", "ankle": "#ef4444"}
_LEVEL_COLORS = {"fp32": "#2266cc", "fp16": "#16a34a", "int8": "#ef4444"}


def latency_timeseries(
    path: str,
    level: str,
    threading: str,
    frames_df: pd.DataFrame,
    k: float,
) -> bool:
    """inferenceMs über frameIndex; Warmup gelb, MAD-Ausreißer rot markiert."""
    if frames_df.empty:
        return False
    df = frames_df.sort_values("frameIndex")
    warm = df["isWarmup"].to_numpy()
    fidx = df["frameIndex"].to_numpy()
    lat = df["inferenceMs"].to_numpy(dtype=float)

    non_warm_lat = lat[~warm]
    _, outlier_mask = mad_filter(non_warm_lat, k=k)
    outlier_full = np.zeros(lat.size, dtype=bool)
    outlier_full[np.where(~warm)[0]] = outlier_mask

    fig, ax = plt.subplots(figsize=(9, 3.5))
    normal = ~warm & ~outlier_full
    ax.plot(fidx, lat, color="#bbbbbb", linewidth=0.6, zorder=1)
    ax.scatter(fidx[normal], lat[normal], s=8, color="#2266cc", label="gültig", zorder=2)
    if warm.any():
        ax.scatter(fidx[warm], lat[warm], s=12, color="#eab308", label="Warmup", zorder=3)
    if outlier_full.any():
        ax.scatter(
            fidx[outlier_full], lat[outlier_full], s=16, color="#ef4444",
            marker="x", label="MAD-Ausreißer", zorder=4,
        )
    ax.set_xlabel("frameIndex")
    ax.set_ylabel("Inferenz [ms]")
    ax.set_title(f"Latenz-Zeitreihe — {level} / {threading}")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def latency_distribution(
    path: str, groups: Sequence[Tuple[str, np.ndarray]]
) -> bool:
    """Boxplot der Latenzen je (Stufe×Threading). groups = [(label, array), ...]."""
    groups = [(lbl, np.asarray(a, dtype=float)) for lbl, a in groups if len(a) > 0]
    if not groups:
        return False
    fig, ax = plt.subplots(figsize=(max(6, len(groups) * 1.2), 4))
    ax.boxplot(
        [a for _, a in groups],
        tick_labels=[lbl for lbl, _ in groups],
        showfliers=True,
    )
    ax.set_ylabel("Inferenz [ms]")
    ax.set_title("Latenzverteilung Stufe × Threading")
    ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def angle_timeseries(
    path: str, level: str, joined: pd.DataFrame, bdc_frames: np.ndarray
) -> bool:
    """Pipeline- vs GT-Kniewinkel über frameIndex, BDCs markiert."""
    if joined.empty:
        return False
    d = joined.sort_values("frameIndex")
    fidx = d["frameIndex"].to_numpy()
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(fidx, d["pipeline_angle"], color="#2266cc", linewidth=1.0, label=f"Pipeline {level}")
    ax.plot(fidx, d["gt_kneeAngle"], color="#111111", linewidth=1.0, linestyle="--", label="GT (Tracker)")
    bdc = np.asarray(bdc_frames, dtype=int)
    for bf in bdc:
        ax.axvline(bf, color="#ef4444", alpha=0.3, linewidth=0.8)
    ax.set_xlabel("frameIndex")
    ax.set_ylabel("Kniewinkel [°]")
    ax.set_title(f"Kniewinkel Pipeline vs GT — {level}")
    handles, _ = ax.get_legend_handles_labels()
    if bdc.size:
        handles.append(
            Line2D([0], [0], color="#ef4444", linewidth=1.2, label="BDC")
        )
    ax.legend(handles=handles, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def bland_altman_plot(path: str, level: str, ba: dict) -> bool:
    """Bland-Altman: Differenz gegen Mittel, Bias + LoA-Linien."""
    mean_pair = np.asarray(ba.get("mean_pair", []), dtype=float)
    diff = np.asarray(ba.get("diff", []), dtype=float)
    if mean_pair.size == 0:
        return False
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    ax.scatter(mean_pair, diff, s=8, color="#2266cc", alpha=0.5)
    ax.axhline(ba["mean_diff"], color="#111111", linewidth=1.0, label=f"Bias {ba['mean_diff']:.2f}°")
    ax.axhline(ba["loa_upper"], color="#ef4444", linestyle="--", linewidth=1.0, label="LoA (±1,96·SD)")
    ax.axhline(ba["loa_lower"], color="#ef4444", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Mittel (Messpipeline, GT) [°]")
    ax.set_ylabel("Differenz Messpipeline − GT [°]")
    ax.set_title(
        f"Bland-Altman: Messpipeline ({level.upper()}) vs "
        "Ground-Truth (Tracker-Software)",
        fontsize=10.5,
    )
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def score_timeseries(
    path: str, level: str, scores_df: pd.DataFrame, threshold: float, names: dict
) -> bool:
    """Keypoint-Scores (hip/knee/ankle) über frameIndex; Schwelle + Warmup grau."""
    if scores_df.empty:
        return False
    d = scores_df.sort_values("frameIndex")
    fidx = d["frameIndex"].to_numpy()
    warm = d["isWarmup"].to_numpy()
    fig, ax = plt.subplots(figsize=(9, 3.5))
    for role in ("hip", "knee", "ankle"):
        ax.plot(
            fidx, d[f"score_{role}"], color=_ROLE_COLORS[role],
            linewidth=1.0, label=names[role],
        )
    ax.axhline(
        threshold, color="#111111", linestyle="--", linewidth=0.8,
        label=f"Schwelle {threshold:.2f}",
    )
    if warm.any():
        wf = fidx[warm]
        ax.axvspan(
            float(wf.min()), float(wf.max()), color="#999999", alpha=0.25,
            label="Warmup",
        )
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("frameIndex")
    ax.set_ylabel("Keypoint-Score")
    ax.set_title(f"Keypoint-Scores — {level}")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def score_hist(
    path: str, level: str, scores_df: pd.DataFrame, threshold: float, names: dict
) -> bool:
    """Score-Histogramme hip/knee/ankle (3 Subplots) mit Schwellen-Marker."""
    if scores_df.empty:
        return False
    roles = ("hip", "knee", "ankle")
    fig, axes = plt.subplots(3, 1, figsize=(6, 7), sharex=True)
    for ax, role in zip(axes, roles):
        s = scores_df[f"score_{role}"].to_numpy(dtype=float)
        ax.hist(s, bins=30, range=(0.0, 1.0), color=_ROLE_COLORS[role], alpha=0.8)
        ax.axvline(threshold, color="#111111", linestyle="--", linewidth=1.0)
        ax.set_ylabel(names[role])
    axes[-1].set_xlabel("Keypoint-Score")
    axes[0].set_title(f"Score-Verteilung — {level}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def delta_vs_score(
    path: str, js_df: pd.DataFrame, ankle_label: str = "lankle"
) -> bool:
    """Scatter |Δθ| gegen Ankle-Score, Farbe je Stufe, globale Regressionsgerade."""
    if js_df.empty:
        return False
    x = js_df["score_ankle"].to_numpy(dtype=float)
    y = js_df["abs_delta"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for level, sub in js_df.groupby("level", sort=False):
        ax.scatter(
            sub["score_ankle"], sub["abs_delta"], s=6, alpha=0.4,
            color=_LEVEL_COLORS.get(level, "#888888"), label=level,
        )
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) >= 2 and np.unique(x[mask]).size >= 2:
        slope, intercept = np.polyfit(x[mask], y[mask], 1)
        xs = np.linspace(float(x[mask].min()), float(x[mask].max()), 50)
        ax.plot(
            xs, slope * xs + intercept, color="#111111", linewidth=1.2,
            label=f"Regression (slope {slope:.1f})",
        )
    ax.set_xlabel(f"{ankle_label}-Score")
    ax.set_ylabel("|θ_pipeline − θ_gt| [°]")
    ax.set_title("Winkelfehler vs. Sprunggelenk-Score")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def delta_hist(path: str, label: str, delta: np.ndarray) -> bool:
    """Histogramm der paarweisen Δθ."""
    d = np.asarray(delta, dtype=float)
    if d.size == 0:
        return False
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(d, bins=40, color="#2266cc", alpha=0.8)
    ax.axvline(float(np.mean(d)), color="#111111", linewidth=1.0, label=f"mean {np.mean(d):.2f}°")
    ax.set_xlabel(f"Δθ {label} [°]")
    ax.set_ylabel("Häufigkeit")
    ax.set_title(f"Paarweise Winkeldifferenz {label}")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True
