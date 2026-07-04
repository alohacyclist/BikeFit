"""Orchestrierung: lädt Sessions + GT, berechnet FF1/FF2/FF3, schreibt alle
Artefakte nach --out. Exit != 0 bei Schema-/Parser-Fehler, n=0 oder
--strict + Frame-Diskrepanz.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from . import accuracy, cycles, gt_tracker, latency, plots, schema
from .angles import SIDE_ANGLE_COLUMN
from .latex import to_latex_decimal_comma

SessionKey = Tuple[str, str, int]  # (level, threading, runIndex)

SUMMARY_COLUMNS = [
    "level", "threading", "runIndex", "n_frames", "n_warmup", "n_outlier",
    "mean_ms", "sd_ms", "median_ms", "mad_ms", "p95_ms", "fps",
    "meets_30fps", "MAE_deg", "RMSE_deg", "meets_rmse_10",
]


def _collect_sessions(args) -> Tuple[List[dict], List[str]]:
    warnings: List[str] = []
    groups = [("fp32", args.fp32), ("fp16", args.fp16), ("int8", args.int8)]
    sessions: List[dict] = []
    seen = set()
    for level, pattern in groups:
        if not pattern:
            continue
        paths = sorted(glob.glob(pattern))
        if not paths:
            warnings.append(f"Kein Treffer für --{level} {pattern!r}.")
        for path in paths:
            if path in seen:
                continue
            seen.add(path)
            session = schema.load_session(path)  # wirft SchemaError
            if session["level"] != level:
                warnings.append(
                    f"{path}: session.level={session['level']} != Flag "
                    f"--{level} (session.level gilt)."
                )
            fl = session.get("_filenameLevel")
            if fl and fl != session["level"]:
                warnings.append(
                    f"{path}: Dateiname-Level {fl} != session.level "
                    f"{session['level']} (session.level gilt)."
                )
            sessions.append(session)
    return sessions, warnings


def _pick_representative(
    keys: List[SessionKey], level: str, prefer: SessionKey | None
) -> SessionKey | None:
    """Wählt für eine Stufe eine repräsentative Session (matcht prefer-Threading
    /-Run, sonst erste vorhandene)."""
    level_keys = [k for k in keys if k[0] == level]
    if not level_keys:
        return None
    if prefer is not None:
        for k in level_keys:
            if k[1] == prefer[1] and k[2] == prefer[2]:
                return k
    return sorted(level_keys)[0]


def _build_ff3_table(agg_lat: pd.DataFrame) -> pd.DataFrame:
    out = agg_lat.copy()
    out["fps"] = out["mean_of_means_ms"].apply(
        lambda m: (1000.0 / m) if m > 0 else 0.0
    )
    out["meets_30fps"] = out["fps"] >= latency.FPS_THRESHOLD
    return out[
        ["level", "threading", "mean_of_means_ms", "sd_of_means_ms",
         "median_of_medians_ms", "p95_of_p95_ms", "fps", "meets_30fps"]
    ]


def run(args) -> int:
    os.makedirs(args.out, exist_ok=True)
    log: Dict = {
        "params": {
            "proband": args.proband, "gt": args.gt, "gt_format": args.gt_format,
            "gt_frame_offset": args.gt_frame_offset, "side": args.side,
            "warmup": args.warmup, "outlier_k": args.outlier_k,
            "strict": args.strict, "out": args.out,
        },
        "warnings": [], "encoding": None, "filters": {}, "bdc_frames": [],
    }

    try:
        sessions, warns = _collect_sessions(args)
    except (schema.SchemaError, OSError, json.JSONDecodeError) as err:
        print(f"FEHLER (Schema/Load): {err}", file=sys.stderr)
        return 2
    log["warnings"].extend(warns)

    if not sessions:
        print("FEHLER: keine Sessions geladen (n=0).", file=sys.stderr)
        return 3

    for s in sessions:
        if s["probandId"] != args.proband:
            log["warnings"].append(
                f"{s['_source']}: probandId {s['probandId']} != "
                f"--proband {args.proband}."
            )
        if s["bodySide"] != args.side:
            log["warnings"].append(
                f"{s['_source']}: bodySide {s['bodySide']} != --side "
                f"{args.side} (--side gewinnt)."
            )
        if s["warmupCount"] != args.warmup:
            log["warnings"].append(
                f"{s['_source']}: warmupCount {s['warmupCount']} != --warmup "
                f"{args.warmup} (isWarmup-Flag ist maßgeblich)."
            )

    fps_values = sorted({int(s["targetFps"]) for s in sessions})
    target_fps = fps_values[0] if fps_values else 30
    if len(fps_values) > 1:
        log["warnings"].append(
            f"Uneinheitliche targetFps {fps_values} — nehme {target_fps}."
        )

    if args.gt_format != "tracker-multi":
        print(
            f"FEHLER: gt-format {args.gt_format!r} nicht unterstützt.",
            file=sys.stderr,
        )
        return 4
    try:
        gt_df, encoding = gt_tracker.parse_tracker_multi(
            args.gt, target_fps=target_fps
        )
    except (gt_tracker.GtParseError, OSError) as err:
        print(f"FEHLER (GT-Parser): {err}", file=sys.stderr)
        return 4
    gt_df = gt_tracker.apply_frame_offset(gt_df, args.gt_frame_offset)
    log["encoding"] = encoding
    n_gt = int(len(gt_df))

    discrepancy = False
    for s in sessions:
        if s["videoTotalFrames"] != n_gt:
            discrepancy = True
            log["warnings"].append(
                f"{s['_source']}: videoTotalFrames {s['videoTotalFrames']} "
                f"!= GT-Zeilen {n_gt}."
            )
    if discrepancy and args.strict:
        print(
            "FEHLER (--strict): Frame-Zahl-Diskrepanz GT vs Pipeline.",
            file=sys.stderr,
        )
        return 5

    side = args.side
    summary_rows = []
    joined_by_key: Dict[SessionKey, pd.DataFrame] = {}
    frames_by_key: Dict[SessionKey, pd.DataFrame] = {}
    pangles_parts = []
    per_session_log = []

    for s in sessions:
        fdf = schema.frames_dataframe(s)
        key: SessionKey = (s["level"], s["threading"], s["runIndex"])
        frames_by_key[key] = fdf

        lat = latency.latency_stats_for_run(fdf, k=args.outlier_k)
        joined = accuracy.join_session(fdf, gt_df, side)
        joined = joined.assign(
            level=s["level"], threading=s["threading"], runIndex=s["runIndex"]
        )
        joined_by_key[key] = joined
        mae, rmse = accuracy.mae_rmse(joined["diff"])

        summary_rows.append(
            {
                "level": s["level"], "threading": s["threading"],
                "runIndex": s["runIndex"],
                "n_frames": lat["n_frames"], "n_warmup": lat["n_warmup"],
                "n_outlier": lat["n_outlier"], "mean_ms": lat["mean_ms"],
                "sd_ms": lat["sd_ms"], "median_ms": lat["median_ms"],
                "mad_ms": lat["mad_ms"], "p95_ms": lat["p95_ms"],
                "fps": lat["fps"], "meets_30fps": lat["meets_30fps"],
                "MAE_deg": mae, "RMSE_deg": rmse,
                "meets_rmse_10": bool(
                    np.isfinite(rmse) and rmse < accuracy.RMSE_THRESHOLD
                ),
            }
        )

        nonwarm = fdf[~fdf["isWarmup"]]
        pangles_parts.append(
            pd.DataFrame(
                {
                    "level": s["level"], "threading": s["threading"],
                    "runIndex": s["runIndex"],
                    "frameIndex": nonwarm["frameIndex"].to_numpy(),
                    "pipeline_angle": nonwarm[SIDE_ANGLE_COLUMN[side]].to_numpy(),
                }
            )
        )
        per_session_log.append(
            {
                "source": os.path.basename(s["_source"]),
                "level": s["level"], "threading": s["threading"],
                "runIndex": s["runIndex"], "n_warmup": lat["n_warmup"],
                "n_outlier": lat["n_outlier"], "n_joined": int(len(joined)),
                "n_dropped": len(s["droppedFrames"]),
            }
        )

    log["filters"]["per_session"] = per_session_log

    summary_df = (
        pd.DataFrame(summary_rows)
        .sort_values(["level", "threading", "runIndex"])
        .reset_index(drop=True)
    )

    # --- Aggregation FF3 (Latenz) + FF1/FF2 (Genauigkeit) ---
    agg_lat = latency.aggregate_latency(summary_df)
    acc_agg = (
        summary_df.groupby(["level", "threading"], sort=False)
        .agg(MAE_deg=("MAE_deg", "mean"), RMSE_deg=("RMSE_deg", "mean"))
        .reset_index()
    )
    summary_aggregated = agg_lat.merge(acc_agg, on=["level", "threading"])[
        ["level", "threading", "mean_of_means_ms", "sd_of_means_ms",
         "MAE_deg", "RMSE_deg"]
    ]

    # --- Paarweise Δθ + Bland-Altman (deskriptiv, gepoolt je Stufe) ---
    pangles_df = (
        pd.concat(pangles_parts, ignore_index=True)
        if pangles_parts else pd.DataFrame(
            columns=["level", "threading", "runIndex", "frameIndex", "pipeline_angle"]
        )
    )
    delta_fp16 = accuracy.pairwise_delta(pangles_df, "fp16", "fp32")
    delta_int8 = accuracy.pairwise_delta(pangles_df, "int8", "fp32")

    ba_by_level: Dict[str, dict] = {}
    joined_by_level: Dict[str, pd.DataFrame] = {}
    for level in schema.LEVELS:
        parts = [j for k, j in joined_by_key.items() if k[0] == level]
        if not parts:
            continue
        pooled = pd.concat(parts, ignore_index=True)
        joined_by_level[level] = pooled
        ba_by_level[level] = accuracy.bland_altman(
            pooled["pipeline_angle"], pooled["gt_kneeAngle"]
        )

    # --- Zyklen (sekundär, deskriptiv): BDC aus FP32, auf alle Stufen ---
    keys = list(joined_by_key.keys())
    fp32_base = _pick_representative(keys, "fp32", prefer=None)
    cycle_summary = pd.DataFrame()
    bdc_per_cycle = pd.DataFrame()
    bdc_frames = np.array([], dtype=int)
    rep_joined_by_level: Dict[str, pd.DataFrame] = {}
    if fp32_base is not None:
        bdc_frames, _, _ = cycles.detect_bdc_frames(joined_by_key[fp32_base])
        log["bdc_frames"] = [int(x) for x in bdc_frames]
        for level in schema.LEVELS:
            rep = _pick_representative(keys, level, prefer=fp32_base)
            if rep is not None:
                rep_joined_by_level[level] = joined_by_key[rep]
        if len(bdc_frames) >= 2 and rep_joined_by_level:
            cycle_summary, bdc_per_cycle = cycles.cycle_metrics(
                bdc_frames, rep_joined_by_level
            )
    cycle_agg = cycles.aggregate_cycles(cycle_summary)

    # --- Artefakte schreiben ---
    _write_outputs(
        args, summary_df, summary_aggregated, agg_lat, acc_agg,
        cycle_summary, bdc_per_cycle, cycle_agg, joined_by_level,
        rep_joined_by_level, frames_by_key, ba_by_level,
        delta_fp16, delta_int8, bdc_frames, log,
    )

    print(f"OK: {len(sessions)} Sessions ausgewertet → {args.out}")
    if log["warnings"]:
        print(f"  ({len(log['warnings'])} Warnung(en), siehe run_log.json)")
    return 0


def _write_outputs(
    args, summary_df, summary_aggregated, agg_lat, acc_agg,
    cycle_summary, bdc_per_cycle, cycle_agg, joined_by_level,
    rep_joined_by_level, frames_by_key, ba_by_level,
    delta_fp16, delta_int8, bdc_frames, log,
) -> None:
    out = args.out

    summary_df.reindex(columns=SUMMARY_COLUMNS).to_csv(
        os.path.join(out, "summary.csv"), index=False
    )
    summary_aggregated.to_csv(
        os.path.join(out, "summary_aggregated.csv"), index=False
    )
    if not cycle_summary.empty:
        cycle_summary.to_csv(os.path.join(out, "cycle_summary.csv"), index=False)
    if not bdc_per_cycle.empty:
        bdc_per_cycle.to_csv(
            os.path.join(out, "bdc_angle_per_cycle.csv"), index=False
        )
    for level, pooled in joined_by_level.items():
        pooled.to_csv(
            os.path.join(out, f"frames_joined_{level}.csv"), index=False
        )

    # Plots
    for (level, threading, _run), fdf in frames_by_key.items():
        # Nur eine repräsentative (erste) Kombination je (level,threading).
        pth = os.path.join(
            out, f"latency_timeseries_{level}_{threading}.png"
        )
        if not os.path.exists(pth):
            plots.latency_timeseries(pth, level, threading, fdf, args.outlier_k)

    boxplot_groups = []
    for (level, threading), grp in summary_df.groupby(
        ["level", "threading"], sort=False
    ):
        lat_all = []
        for _run in grp["runIndex"]:
            fdf = frames_by_key.get((level, threading, int(_run)))
            if fdf is None:
                continue
            nonwarm = fdf[~fdf["isWarmup"]]
            kept, _ = latency.mad_filter(
                nonwarm["inferenceMs"].to_numpy(), k=args.outlier_k
            )
            lat_all.append(kept)
        if lat_all:
            boxplot_groups.append(
                (f"{level}/{threading}", np.concatenate(lat_all))
            )
    plots.latency_distribution(
        os.path.join(out, "latency_distribution.png"), boxplot_groups
    )

    for level, joined in rep_joined_by_level.items():
        plots.angle_timeseries(
            os.path.join(out, f"angle_timeseries_{level}.png"),
            level, joined, bdc_frames,
        )
    for level, ba in ba_by_level.items():
        plots.bland_altman_plot(
            os.path.join(out, f"bland_altman_{level}_vs_gt.png"), level, ba
        )
    plots.delta_hist(
        os.path.join(out, "delta_hist_fp16_vs_fp32.png"),
        "FP16 − FP32", delta_fp16,
    )
    plots.delta_hist(
        os.path.join(out, "delta_hist_int8_vs_fp32.png"),
        "INT8 − FP32", delta_int8,
    )

    _write_latex(
        out, summary_aggregated, agg_lat, ba_by_level, delta_fp16, delta_int8,
        cycle_agg,
    )

    with open(os.path.join(out, "run_log.json"), "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=2, ensure_ascii=False)


def _write_latex(
    out, summary_aggregated, agg_lat, ba_by_level, delta_fp16, delta_int8,
    cycle_agg,
) -> None:
    # FF1 — FP32-Baseline vs GT.
    ff1 = summary_aggregated[summary_aggregated["level"] == "fp32"][
        ["threading", "MAE_deg", "RMSE_deg"]
    ].copy()
    ff1["akzeptanz_rmse<10"] = ff1["RMSE_deg"] < accuracy.RMSE_THRESHOLD
    _save_tex(
        out, "table_ff1.tex",
        to_latex_decimal_comma(
            ff1, "FF1 — FP32-Baseline gegen Tracker-Referenz (MAE, RMSE).",
            "tab:ff1",
        ),
    )

    # FF2 — Quantisierungseffekt (MAE/RMSE je Stufe + Δθ + Bland-Altman).
    ff2_rows = []
    acc_by_level = (
        summary_aggregated.groupby("level", sort=False)
        .agg(MAE_deg=("MAE_deg", "mean"), RMSE_deg=("RMSE_deg", "mean"))
        .reset_index()
    )
    acc_map = {r["level"]: r for _, r in acc_by_level.iterrows()}
    d_fp16 = accuracy.delta_stats(delta_fp16)
    d_int8 = accuracy.delta_stats(delta_int8)
    delta_map = {"fp16": d_fp16, "int8": d_int8}
    for level in schema.LEVELS:
        if level not in acc_map:
            continue
        ba = ba_by_level.get(level, {})
        dstat = delta_map.get(level, {})
        ff2_rows.append(
            {
                "level": level,
                "MAE_deg": acc_map[level]["MAE_deg"],
                "RMSE_deg": acc_map[level]["RMSE_deg"],
                "delta_mean_deg": dstat.get("mean", float("nan")),
                "delta_sd_deg": dstat.get("sd", float("nan")),
                "ba_bias_deg": ba.get("mean_diff", float("nan")),
                "ba_loa_low": ba.get("loa_lower", float("nan")),
                "ba_loa_high": ba.get("loa_upper", float("nan")),
            }
        )
    _save_tex(
        out, "table_ff2.tex",
        to_latex_decimal_comma(
            pd.DataFrame(ff2_rows),
            "FF2 — Quantisierungseffekt: MAE/RMSE, paarweise Δθ (vs FP32), "
            "Bland-Altman (Bias, LoA).",
            "tab:ff2",
        ),
    )

    # FF2 Zyklus-Ebene (sekundär, deskriptiv).
    if not cycle_agg.empty:
        _save_tex(
            out, "table_ff2_cycle.tex",
            to_latex_decimal_comma(
                cycle_agg,
                "FF2 (Zyklus-Ebene, deskriptiv) — MAE/RMSE über Pedalzyklen.",
                "tab:ff2cycle",
            ),
        )

    # FF3 — Latenz.
    ff3 = _build_ff3_table(agg_lat)
    _save_tex(
        out, "table_ff3.tex",
        to_latex_decimal_comma(
            ff3,
            "FF3 — Inferenzlatenz je Stufe×Threading (mean, SD, Median, P95, "
            "FPS; Schwelle 30 fps).",
            "tab:ff3",
        ),
    )


def _save_tex(out: str, name: str, content: str) -> None:
    with open(os.path.join(out, name), "w", encoding="utf-8") as fh:
        fh.write(content)
