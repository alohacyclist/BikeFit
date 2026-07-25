#!/usr/bin/env python3
"""Tests für das Post-Processing-Package (Vertrag v1.0.0).

Läuft mit stdlib-unittest (kein pytest nötig):
    python3 scripts/test_postprocess_benchmark.py
    python3 -m unittest discover -s scripts
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from postprocess import accuracy, angles, cycles, gt_tracker, latency, schema  # noqa: E402
from postprocess import keypoint_quality as kq  # noqa: E402
from postprocess.cli import build_parser  # noqa: E402
from postprocess.report import run as report_run  # noqa: E402

KP_COUNT = schema.KEYPOINT_COUNT


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _knee_from_angle(target_deg: float):
    """Hüfte/Knie/Sprunggelenk, deren Innenwinkel == target_deg ergibt."""
    rad = math.radians(target_deg)
    hip = (0.0, 100.0)
    knee = (0.0, 0.0)
    ankle = (100.0 * math.sin(rad), 100.0 * math.cos(rad))
    return hip, knee, ankle


def make_frame(i: int, angle_left: float, inference_ms: float, is_warmup: bool):
    return {
        "frameIndex": i,
        "timestampMs": float(i * 33),
        "inferenceMs": inference_ms,
        "fps": (1000.0 / inference_ms) if inference_ms > 0 else 0.0,
        "kneeAngleRight": 95.0,
        "kneeAngleLeft": angle_left,
        "keypoints": [{"x": 0.0, "y": 0.0} for _ in range(KP_COUNT)],
        "keypointScores": [0.9] * KP_COUNT,
        "isWarmup": is_warmup,
    }


def make_session(level="fp32", threading="single", run_index=0, n=120, warmup=5):
    frames = []
    for i in range(n):
        angle = 120.0 + 40.0 * math.sin(2 * math.pi * 4 * i / n)
        frames.append(make_frame(i, angle, 5.0 + 0.4 * math.sin(i), i < warmup))
    return {
        "schemaVersion": "1.0.0",
        "probandId": "P01",
        "level": level,
        "runIndex": run_index,
        "createdAt": "2026-07-04T12:00:00.000Z",
        "targetFps": 30,
        "bodySide": "left",
        "threading": threading,
        "videoDurationSec": n / 30.0,
        "videoTotalFrames": n,
        "warmupCount": warmup,
        "modelFingerprintSha256": "abc123",
        "modelUrl": f"/models/movenet-lightning-{level}.tflite",
        "modelLoadMs": 42.0,
        "userAgent": "test-agent",
        "hardware": {"device": "TestBook", "cpu": "T-CPU", "os": "TestOS", "browser": "T 1"},
        "frames": frames,
        "droppedFrames": [],
    }


def write_session(directory, session):
    stamp = "20260704T120000Z"
    name = f"benchmark_{session['probandId']}_{session['level']}_{stamp}.json"
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(session, fh)
    return path


def make_environment():
    """Minimaler v1.1.0-environment-Block (wie vom Browser-Exporter erzeugt)."""
    return {
        "capturedAt": "2026-07-04T12:01:00.000Z",
        "crossOriginIsolated": True,
        "hardwareConcurrency": 8,
        "deviceMemoryGb": 8,
        "numThreads": 4,
        "threadingMode": "multi",
        "tfjs": {
            "core": "4.22.0", "backendWasm": "4.22.0",
            "backendWebgl": "4.22.0", "converter": "4.22.0",
            "tflite": "0.0.1-alpha.9", "poseDetection": "2.1.3",
        },
        "userAgentData": {
            "platform": "macOS", "platformVersion": "14.5.0",
            "uaFullVersion": "126.0.0.0", "model": "",
            "architecture": "arm", "bitness": "64",
        },
        "battery": {"level": 1.0, "charging": True},
        "power": {"acPower": True, "lowPowerModeOff": True,
                  "batteryPercentNote": 100},
    }


def write_tracker(path, n=120, bias=2.0):
    """Tracker-Multi-Datei: 13 Felder, Delimiter '.', Dezimalkomma."""
    lines = ["#multi:", ".Masse A....Masse B....Masse C....",
             ".t.x.y.θ.frame.x.y.θ.frame.x.y.θ.frame."]
    for i in range(n):
        frame = i + 1
        t = (frame - 1) / 30.0
        angle = 120.0 + 40.0 * math.sin(2 * math.pi * 4 * i / n) + bias
        hip, knee, ankle = _knee_from_angle(angle)
        vals = [t, hip[0], hip[1], 0.0, frame,
                knee[0], knee[1], 0.0, frame,
                ankle[0], ankle[1], 0.0, frame]
        toks = []
        for k, v in enumerate(vals):
            if k in (4, 8, 12):  # frame-Felder ganzzahlig
                toks.append(str(int(v)))
            else:
                toks.append(f"{v:.5f}".replace(".", ","))
        lines.append(".".join(toks))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


# --------------------------------------------------------------------------- #
# angles
# --------------------------------------------------------------------------- #
class AnglesTest(unittest.TestCase):
    def test_straight_leg_is_180(self):
        a = angles.knee_angle((0, 200), (0, 100), (0, 0))
        self.assertAlmostEqual(a, 180.0, places=6)

    def test_right_angle(self):
        a = angles.knee_angle((0, 100), (0, 0), (100, 0))
        self.assertAlmostEqual(a, 90.0, places=6)

    def test_reflection_invariance(self):
        up = angles.knee_angle((0, 100), (0, 0), (50, 50))
        down = angles.knee_angle((0, -100), (0, 0), (50, -50))
        self.assertAlmostEqual(up, down, places=9)

    def test_reconstruct_target(self):
        hip, knee, ankle = _knee_from_angle(137.0)
        self.assertAlmostEqual(angles.knee_angle(hip, knee, ankle), 137.0, places=5)


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #
class SchemaTest(unittest.TestCase):
    def test_valid_passes(self):
        schema.validate_session(make_session())

    def test_bad_schema_version(self):
        s = make_session()
        s["schemaVersion"] = "0.9.0"
        with self.assertRaises(schema.SchemaError):
            schema.validate_session(s)

    def test_missing_key(self):
        s = make_session()
        del s["hardware"]
        with self.assertRaises(schema.SchemaError):
            schema.validate_session(s)

    def test_invariant_violation(self):
        s = make_session()
        s["videoTotalFrames"] = 999
        with self.assertRaises(schema.SchemaError):
            schema.validate_session(s)

    def test_keypoints_wrong_length(self):
        s = make_session()
        s["frames"][0]["keypoints"] = [{"x": 0, "y": 0}]
        with self.assertRaises(schema.SchemaError):
            schema.validate_session(s)

    def test_bad_drop_reason(self):
        s = make_session()
        s["droppedFrames"] = [{"frameIndex": 3, "reason": "banana"}]
        s["videoTotalFrames"] = len(s["frames"]) + 1
        with self.assertRaises(schema.SchemaError):
            schema.validate_session(s)

    def test_dropped_counts_into_invariant(self):
        s = make_session()
        s["droppedFrames"] = [{"frameIndex": 3, "reason": "seek_timeout"}]
        s["videoTotalFrames"] = len(s["frames"]) + 1
        schema.validate_session(s)  # darf nicht werfen

    def test_schema_11_with_environment(self):
        s = make_session()
        s["schemaVersion"] = "1.1.0"
        s["environment"] = make_environment()
        schema.validate_session(s)  # kein Raise

    def test_schema_environment_not_object(self):
        s = make_session()
        s["schemaVersion"] = "1.1.0"
        s["environment"] = "nope"
        with self.assertRaises(schema.SchemaError):
            schema.validate_session(s)

    def test_unsupported_schema_version(self):
        s = make_session()
        s["schemaVersion"] = "2.0.0"
        with self.assertRaises(schema.SchemaError):
            schema.validate_session(s)

    def test_filename_level_parse(self):
        self.assertEqual(
            schema.parse_level_from_filename("benchmark_P01_int8_X.json"), "int8"
        )
        self.assertIsNone(schema.parse_level_from_filename("random.json"))


# --------------------------------------------------------------------------- #
# gt_tracker
# --------------------------------------------------------------------------- #
class GtTrackerTest(unittest.TestCase):
    def test_parse_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_tracker(os.path.join(tmp, "gt.txt"), n=30, bias=0.0)
            df, enc = gt_tracker.parse_tracker_multi(path, target_fps=30)
        self.assertEqual(len(df), 30)
        self.assertEqual(int(df["frame_gt"].iloc[0]), 1)
        # Winkel rekonstruiert (bias 0): erster Frame 120 + 0 = 120°
        self.assertAlmostEqual(df["gt_kneeAngle"].iloc[0], 120.0, places=3)

    def test_frame_offset(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_tracker(os.path.join(tmp, "gt.txt"), n=10)
            df, _ = gt_tracker.parse_tracker_multi(path)
        df = gt_tracker.apply_frame_offset(df, -1)
        self.assertEqual(int(df["frameIndex"].iloc[0]), 0)

    def test_wrong_field_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "bad.txt")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("#multi:\nheader2\nheader3\n1,0.2,0.3,0\n")
            with self.assertRaises(gt_tracker.GtParseError):
                gt_tracker.parse_tracker_multi(p)

    def test_drift_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "drift.txt")
            # t passt nicht zu (frame-1)/30 -> Drift-Fehler
            row = ".".join(["9,99999"] + ["1,0"] * 3 + ["1"] + ["1,0"] * 3
                           + ["1"] + ["1,0"] * 3 + ["1"])
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(f"#multi:\nh2\nh3\n{row}\n")
            with self.assertRaises(gt_tracker.GtParseError):
                gt_tracker.parse_tracker_multi(p)


# --------------------------------------------------------------------------- #
# latency
# --------------------------------------------------------------------------- #
class LatencyTest(unittest.TestCase):
    def test_nearest_rank_p95(self):
        vals = list(range(1, 101))  # 1..100
        self.assertEqual(latency.nearest_rank_p95(vals), 95.0)

    def test_nearest_rank_single(self):
        self.assertEqual(latency.nearest_rank_p95([7.0]), 7.0)

    def test_mad_filter_removes_outlier(self):
        # Realistische Jitter-Basis (MAD>0) + ein klarer Spike.
        vals = [9.8, 10.2, 9.9, 10.1, 10.0] * 4 + [1000.0]
        kept, outlier = latency.mad_filter(vals, k=3.0)
        self.assertEqual(int(outlier.sum()), 1)
        self.assertNotIn(1000.0, list(kept))

    def test_mad_zero_no_filter(self):
        # MAD==0 (alle identisch): bewusst KEIN Filter — verhindert Über-
        # Filterung bei grob quantisiertem performance.now().
        vals = [5.0] * 10
        kept, outlier = latency.mad_filter(vals, k=3.0)
        self.assertEqual(int(outlier.sum()), 0)
        self.assertEqual(len(kept), 10)

    def test_stats_population_sd(self):
        import pandas as pd
        frames = pd.DataFrame(
            {
                "inferenceMs": [10.0, 20.0, 30.0, 40.0],
                "isWarmup": [False, False, False, False],
            }
        )
        st = latency.latency_stats_for_run(frames, k=3.0)
        self.assertAlmostEqual(st["mean_ms"], 25.0)
        self.assertAlmostEqual(st["sd_ms"], math.sqrt(125.0), places=9)
        self.assertAlmostEqual(st["fps"], 40.0)

    def test_warmup_excluded(self):
        import pandas as pd
        frames = pd.DataFrame(
            {
                "inferenceMs": [999.0, 999.0, 10.0, 20.0],
                "isWarmup": [True, True, False, False],
            }
        )
        st = latency.latency_stats_for_run(frames)
        self.assertEqual(st["n_warmup"], 2)
        self.assertAlmostEqual(st["mean_ms"], 15.0)


# --------------------------------------------------------------------------- #
# accuracy
# --------------------------------------------------------------------------- #
class AccuracyTest(unittest.TestCase):
    def test_mae_rmse(self):
        mae, rmse = accuracy.mae_rmse([3.0, -4.0])
        self.assertAlmostEqual(mae, 3.5)
        self.assertAlmostEqual(rmse, math.sqrt(12.5))

    def test_bland_altman(self):
        ba = accuracy.bland_altman([10.0, 12.0, 14.0], [9.0, 11.0, 13.0])
        self.assertAlmostEqual(ba["mean_diff"], 1.0)
        self.assertEqual(ba["n"], 3)

    def test_pairwise_delta(self):
        import pandas as pd
        df = pd.DataFrame(
            {
                "level": ["fp32", "fp32", "fp16", "fp16"],
                "threading": ["single"] * 4,
                "runIndex": [0, 0, 0, 0],
                "frameIndex": [0, 1, 0, 1],
                "pipeline_angle": [100.0, 110.0, 101.0, 108.0],
            }
        )
        delta = accuracy.pairwise_delta(df, "fp16", "fp32")
        self.assertEqual(sorted(delta.tolist()), [-2.0, 1.0])


# --------------------------------------------------------------------------- #
# cycles
# --------------------------------------------------------------------------- #
class CyclesTest(unittest.TestCase):
    def test_detect_bdc(self):
        import pandas as pd
        n = 120
        rows = []
        for i in range(n):
            ang = 120.0 + 40.0 * math.sin(2 * math.pi * 4 * i / n)
            rows.append({"frameIndex": i, "pipeline_angle": ang, "gt_kneeAngle": ang})
        joined = pd.DataFrame(rows)
        bdc, _, _ = cycles.detect_bdc_frames(joined)
        # 4 Sinus-Perioden -> ~4 Maxima
        self.assertGreaterEqual(len(bdc), 3)


# --------------------------------------------------------------------------- #
# Integration (report.run end-to-end)
# --------------------------------------------------------------------------- #
class KeypointQualityTest(unittest.TestCase):
    def test_frame_scores(self):
        s = make_session(n=10, warmup=2)
        df = schema.frame_scores(s, {"hip": 11, "knee": 13, "ankle": 15})
        self.assertEqual(
            list(df.columns),
            ["frameIndex", "isWarmup", "score_hip", "score_knee", "score_ankle"],
        )
        self.assertEqual(len(df), 10)
        self.assertTrue((df["score_ankle"] == 0.9).all())
        self.assertEqual(int(df["isWarmup"].sum()), 2)

    def test_score_validation_rejects_nonnumber(self):
        s = make_session(n=3, warmup=0)
        s["frames"][0]["keypointScores"][5] = "x"
        with self.assertRaises(schema.SchemaError):
            schema.validate_session(s)

    def test_keypoint_quality_rows(self):
        import pandas as pd
        sdf = pd.DataFrame({
            "frameIndex": range(4), "isWarmup": [False] * 4,
            "score_hip": [0.9, 0.9, 0.9, 0.9],
            "score_knee": [0.8, 0.8, 0.8, 0.8],
            "score_ankle": [0.1, 0.2, 0.6, 0.7],  # 2 < 0.5
        })
        rows = kq.keypoint_quality_rows("int8", "multi", 0, sdf, "left", 0.5)
        self.assertEqual(len(rows), 3)
        by_name = {r["keypoint_name"]: r for r in rows}
        self.assertEqual(set(by_name), {"lhip", "lknee", "lankle"})
        self.assertEqual(by_name["lankle"]["keypoint_idx"], 15)
        self.assertEqual(by_name["lankle"]["n_frames_low_conf"], 2)
        self.assertAlmostEqual(by_name["lankle"]["pct_frames_low_conf"], 50.0)
        self.assertEqual(by_name["lhip"]["n_frames_low_conf"], 0)

    def test_session_score_summary(self):
        import pandas as pd
        sdf = pd.DataFrame({
            "frameIndex": range(2), "isWarmup": [False, False],
            "score_hip": [0.5, 0.7], "score_knee": [0.4, 0.6],
            "score_ankle": [0.2, 0.8],  # 1 < 0.5 -> 50%
        })
        out = kq.session_score_summary(sdf, "left", 0.5)
        self.assertAlmostEqual(out["lhip_mean_score"], 0.6)
        self.assertAlmostEqual(out["lankle_mean_score"], 0.5)
        self.assertAlmostEqual(out["pct_lankle_below_0_5"], 50.0)

    def test_summary_score_columns(self):
        self.assertEqual(
            kq.summary_score_columns("left"),
            ["lhip_mean_score", "lknee_mean_score", "lankle_mean_score",
             "pct_lankle_below_0_5"],
        )
        self.assertEqual(kq.summary_score_columns("right")[-1],
                         "pct_rankle_below_0_5")

    def test_weighted_mae(self):
        self.assertAlmostEqual(kq.weighted_mae([2.0, 4.0], [1.0, 3.0]), 3.5)
        self.assertTrue(math.isnan(kq.weighted_mae([2.0, 4.0], [0.0, 0.0])))
        self.assertTrue(math.isnan(kq.weighted_mae([], [])))

    def test_pearson_r(self):
        self.assertAlmostEqual(kq.pearson_r([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertTrue(math.isnan(kq.pearson_r([1, 1, 1], [1, 2, 3])))
        self.assertTrue(math.isnan(kq.pearson_r([1.0], [2.0])))

    def test_worst_case_frame(self):
        import pandas as pd
        df = pd.DataFrame({
            "frameIndex": [10, 11, 12],
            "abs_delta": [30.0, 5.0, 40.0],
            "score_ankle": [0.1, 0.2, 0.9],  # frame 12 hat höchstes Δ, aber Score 0.9
            "threading": ["multi"] * 3, "runIndex": [0, 0, 0],
        })
        wc = kq.worst_case_frame(df, ankle_max=0.3)
        self.assertEqual(wc["frameIndex"], 10)  # max |Δ| UNTER Schwelle
        self.assertAlmostEqual(wc["lankle_score"], 0.1)
        self.assertIn("lankle_score", wc)  # left-Default
        wc_r = kq.worst_case_frame(df, ankle_max=0.3, side="right")
        self.assertIn("rankle_score", wc_r)  # right-side Key
        self.assertNotIn("lankle_score", wc_r)
        none = kq.worst_case_frame(
            df.assign(score_ankle=[0.5, 0.6, 0.9]), ankle_max=0.3
        )
        self.assertIsNone(none)


class ReportIntegrationTest(unittest.TestCase):
    def _args(self, tmp, exports, out, **over):
        argv = [
            "--proband", "P01",
            "--gt", os.path.join(tmp, "gt.txt"),
            "--gt-format", "tracker-multi",
            "--gt-frame-offset", "-1",
            "--fp32", os.path.join(exports, "benchmark_P01_fp32_*.json"),
            "--fp16", os.path.join(exports, "benchmark_P01_fp16_*.json"),
            "--int8", os.path.join(exports, "benchmark_P01_int8_*.json"),
            "--side", "left", "--warmup", "5", "--outlier-k", "3.0",
            "--out", out,
        ]
        for k, v in over.items():
            argv += [f"--{k}", str(v)] if v is not True else [f"--{k}"]
        return build_parser().parse_args(argv)

    def _setup(self, tmp):
        exports = os.path.join(tmp, "exports")
        os.makedirs(exports)
        for level in ("fp32", "fp16", "int8"):
            write_session(exports, make_session(level=level))
        write_tracker(os.path.join(tmp, "gt.txt"), n=120)
        return exports

    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            exports = self._setup(tmp)
            out = os.path.join(tmp, "results")
            rc = report_run(self._args(tmp, exports, out))
            self.assertEqual(rc, 0)
            for name in ("summary.csv", "summary_aggregated.csv",
                         "run_log.json", "table_ff1.tex", "table_ff2.tex",
                         "table_ff3.tex", "latency_distribution.png",
                         "keypoint_quality.csv", "delta_vs_score.png",
                         "score_timeseries_fp32.png", "score_hist_fp32.png"):
                self.assertTrue(
                    os.path.exists(os.path.join(out, name)), f"fehlt: {name}"
                )
            import pandas as pd
            report_mod = __import__("postprocess.report", fromlist=["x"])
            summary = pd.read_csv(os.path.join(out, "summary.csv"))
            self.assertEqual(
                list(summary.columns),
                report_mod.SUMMARY_COLUMNS + kq.summary_score_columns("left"),
            )
            self.assertEqual(len(summary), 3)  # fp32/fp16/int8 je 1 Lauf

            agg = pd.read_csv(os.path.join(out, "summary_aggregated.csv"))
            self.assertIn("MAE_weighted", agg.columns)
            kqdf = pd.read_csv(os.path.join(out, "keypoint_quality.csv"))
            self.assertEqual(len(kqdf), 9)  # 3 Stufen × 3 Keypoints
            self.assertEqual(
                set(kqdf["keypoint_name"]), {"lhip", "lknee", "lankle"}
            )
            with open(os.path.join(out, "run_log.json"), encoding="utf-8") as fh:
                rlog = json.load(fh)
            self.assertIn("delta_score_correlation", rlog)
            self.assertIn("worst_case_frames", rlog)
            self.assertEqual(rlog["params"]["score_threshold"], 0.5)
            # Pearson r ONLY, kein p_value (bewusste Entscheidung).
            corr_fp32 = rlog["delta_score_correlation"]["fp32"]
            self.assertEqual(set(corr_fp32), {"pearson_r"})
            self.assertNotIn("p_value", corr_fp32)
            # 1.0.0-Altdaten ohne environment -> keine environment.csv.
            self.assertFalse(
                os.path.exists(os.path.join(out, "environment.csv"))
            )

    def test_end_to_end_environment_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            exports = os.path.join(tmp, "exports")
            os.makedirs(exports)
            for level in ("fp32", "fp16", "int8"):
                s = make_session(level=level)
                s["schemaVersion"] = "1.1.0"
                s["environment"] = make_environment()
                write_session(exports, s)
            write_tracker(os.path.join(tmp, "gt.txt"), n=120)
            out = os.path.join(tmp, "results")
            rc = report_run(self._args(tmp, exports, out))
            self.assertEqual(rc, 0)
            import pandas as pd
            report_mod = __import__("postprocess.report", fromlist=["x"])
            envp = os.path.join(out, "environment.csv")
            self.assertTrue(os.path.exists(envp))
            envdf = pd.read_csv(envp)
            self.assertEqual(
                list(envdf.columns), report_mod.ENVIRONMENT_COLUMNS
            )
            self.assertEqual(len(envdf), 3)
            self.assertTrue((envdf["numThreads"] == 4).all())
            self.assertTrue(bool(envdf["crossOriginIsolated"].all()))

    def test_no_sessions_exit3(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "exports"))
            write_tracker(os.path.join(tmp, "gt.txt"), n=10)
            args = self._args(tmp, os.path.join(tmp, "exports"),
                              os.path.join(tmp, "out"))
            self.assertEqual(report_run(args), 3)

    def test_bad_gt_exit4(self):
        with tempfile.TemporaryDirectory() as tmp:
            exports = self._setup(tmp)
            with open(os.path.join(tmp, "gt.txt"), "w", encoding="utf-8") as fh:
                fh.write("#multi:\nh2\nh3\ngarbage,line,here\n")
            rc = report_run(self._args(tmp, exports, os.path.join(tmp, "out")))
            self.assertEqual(rc, 4)

    def test_strict_discrepancy_exit5(self):
        with tempfile.TemporaryDirectory() as tmp:
            exports = self._setup(tmp)
            # GT hat weniger Zeilen als videoTotalFrames -> Diskrepanz
            write_tracker(os.path.join(tmp, "gt.txt"), n=100)
            args = self._args(tmp, exports, os.path.join(tmp, "out"), strict=True)
            self.assertEqual(report_run(args), 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
