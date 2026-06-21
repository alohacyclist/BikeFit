#!/usr/bin/env python3
"""Tests für postprocess_benchmark.

Dummy-Daten spiegeln exakt das Rohformat aus dem aktuellen Code:
  - FrameMeasurement: src/services/BenchmarkExporter.ts (frameIndex, timestampMs,
    inferenceMs, fps, kneeAngleRight, kneeAngleLeft, keypointScores[17], isWarmup)
  - Session-Metadaten inkl. targetFps/durationSeconds/expectedFrames/warmupFrames

Lauf: python3 scripts/test_postprocess_benchmark.py
   bzw. python3 -m unittest discover -s scripts
"""

import json
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import postprocess_benchmark as pp  # noqa: E402

KEYPOINT_SCORES = [0.9] * pp.KEYPOINT_COUNT


def make_frame(
    index: int,
    inference_ms: float,
    is_warmup: bool,
    knee_right=None,
    knee_left=None,
    fps: float = 0.0,
):
    """Ein Frame im exakten Export-Format."""
    return {
        "frameIndex": index,
        "timestampMs": float(index),
        "inferenceMs": inference_ms,
        "fps": fps,
        "kneeAngleRight": knee_right,
        "kneeAngleLeft": knee_left,
        "keypointScores": list(KEYPOINT_SCORES),
        "isWarmup": is_warmup,
    }


def make_session(frames, **overrides):
    """Eine Session im exakten Export-Format (Metadaten überschreibbar)."""
    session = {
        "participantId": "P01",
        "quantizationLevel": "fp32",
        "startTimestamp": "2026-06-21T10:00:00.000Z",
        "systemInfo": {
            "userAgent": "test",
            "hardwareConcurrency": 8,
            "crossOriginIsolated": True,
            "sharedArrayBufferAvailable": True,
        },
        "warmupFrames": 5,
        "lockedSide": "right",
        "modelFingerprint": {"sha256": "abc", "sizeBytes": 1234},
        "threadingMode": "single",
        "numThreads": 1,
        "videoSource": "file",
        "videoSourceName": "clip.mp4",
        "targetFps": 30,
        "durationSeconds": 2.0,
        "expectedFrames": 60,
        "frames": frames,
    }
    session.update(overrides)
    return session


def default_session():
    """5 Warmup-Frames (hohe Latenz, müssen ignoriert werden) + 4 valide.

    Valide inferenceMs = [10, 20, 30, 40]:
      mean = 25.0, pop-SD = sqrt(125), p50(nearest-rank, n=4) = 30, p95 = 40,
      fps = 1000/25 = 40.0
    Kniewinkel rechts valide = [140, 150, None, 160] -> mean 150.0
    Kniewinkel links valide = alle None -> None
    """
    warmup = [make_frame(i + 1, 1000.0, True) for i in range(5)]
    valid = [
        make_frame(6, 10.0, False, knee_right=140.0, knee_left=None),
        make_frame(7, 20.0, False, knee_right=150.0, knee_left=None),
        make_frame(8, 30.0, False, knee_right=None, knee_left=None),
        make_frame(9, 40.0, False, knee_right=160.0, knee_left=None),
    ]
    return make_session(warmup + valid)


class SummarizeTest(unittest.TestCase):
    def setUp(self):
        self.summary = pp.summarize(default_session())

    def test_frame_counts_exclude_warmup(self):
        self.assertEqual(self.summary["totalFrames"], 9)
        self.assertEqual(self.summary["validFrames"], 4)

    def test_mean_latency_ignores_warmup(self):
        # Würden Warmup (1000ms) mitzählen, läge der Mittelwert weit höher.
        self.assertAlmostEqual(self.summary["meanInferenceMs"], 25.0)

    def test_population_std(self):
        self.assertAlmostEqual(
            self.summary["stdInferenceMs"], math.sqrt(125.0), places=12
        )

    def test_nearest_rank_percentiles(self):
        self.assertEqual(self.summary["p50Ms"], 30.0)
        self.assertEqual(self.summary["p95Ms"], 40.0)

    def test_fps_is_throughput_of_mean_latency(self):
        # ANPASSUNG: 1000 / mean_latency, NICHT Mittel der per-Frame-fps.
        self.assertAlmostEqual(self.summary["fps"], 40.0)

    def test_knee_means_handle_nulls(self):
        self.assertAlmostEqual(self.summary["meanKneeAngleRight"], 150.0)
        self.assertIsNone(self.summary["meanKneeAngleLeft"])

    def test_metadata_passthrough(self):
        self.assertEqual(self.summary["quantizationLevel"], "fp32")
        self.assertEqual(self.summary["targetFps"], 30)
        self.assertEqual(self.summary["durationSeconds"], 2.0)
        self.assertEqual(self.summary["expectedFrames"], 60)


class EdgeCaseTest(unittest.TestCase):
    def test_all_warmup_yields_zeroed_stats(self):
        session = make_session([make_frame(i + 1, 50.0, True) for i in range(3)])
        summary = pp.summarize(session)
        self.assertEqual(summary["totalFrames"], 3)
        self.assertEqual(summary["validFrames"], 0)
        self.assertEqual(summary["meanInferenceMs"], 0.0)
        self.assertEqual(summary["fps"], 0.0)
        self.assertIsNone(summary["meanKneeAngleRight"])

    def test_single_valid_frame_std_zero(self):
        session = make_session(
            [make_frame(1, 12.0, False, knee_right=100.0)]
        )
        summary = pp.summarize(session)
        self.assertEqual(summary["validFrames"], 1)
        self.assertAlmostEqual(summary["meanInferenceMs"], 12.0)
        self.assertEqual(summary["stdInferenceMs"], 0.0)
        self.assertAlmostEqual(summary["fps"], 1000.0 / 12.0)

    def test_nearest_rank_empty(self):
        self.assertEqual(pp.nearest_rank([], 50), 0.0)


class ValidationTest(unittest.TestCase):
    def test_non_dict_session(self):
        with self.assertRaises(ValueError):
            pp.validate_session([1, 2, 3])

    def test_missing_frames(self):
        with self.assertRaises(ValueError):
            pp.validate_session({"participantId": "P01"})

    def test_empty_frames(self):
        with self.assertRaises(ValueError):
            pp.validate_session(make_session([]))

    def test_frame_missing_inference_ms(self):
        bad = {"frameIndex": 1, "isWarmup": False}
        with self.assertRaises(ValueError):
            pp.validate_session(make_session([bad]))

    def test_inference_ms_not_number(self):
        bad = make_frame(1, 10.0, False)
        bad["inferenceMs"] = "fast"
        with self.assertRaises(ValueError):
            pp.validate_session(make_session([bad]))

    def test_is_warmup_not_bool(self):
        bad = make_frame(1, 10.0, False)
        bad["isWarmup"] = 0
        with self.assertRaises(ValueError):
            pp.validate_session(make_session([bad]))

    def test_knee_angle_wrong_type(self):
        bad = make_frame(1, 10.0, False)
        bad["kneeAngleRight"] = "straight"
        with self.assertRaises(ValueError):
            pp.validate_session(make_session([bad]))

    def test_keypoint_scores_not_list(self):
        bad = make_frame(1, 10.0, False)
        bad["keypointScores"] = "x"
        with self.assertRaises(ValueError):
            pp.validate_session(make_session([bad]))

    def test_valid_session_passes(self):
        pp.validate_session(default_session())  # darf nicht werfen


class LoadSessionTest(unittest.TestCase):
    def test_roundtrip_from_file(self):
        session = default_session()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(session, handle)
            loaded = pp.load_session(path)
        summary = pp.summarize(loaded)
        self.assertEqual(summary["validFrames"], 4)
        self.assertAlmostEqual(summary["meanInferenceMs"], 25.0)

    def test_malformed_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"frames": []}, handle)
            with self.assertRaises(ValueError):
                pp.load_session(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
