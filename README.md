# BikeFit

A browser-only measurement app from my bachelor's thesis. It measures the knee angle on
a bike from video with MoveNet (SinglePose Lightning) via TensorFlow.js TFLite on the
WASM backend, and compares three post-training quantization levels of the model:
FP32, FP16 and INT8. Accuracy is validated frame by frame against a 2D video-analysis
tool as ground truth; latency is measured per inference.

It is a research tool, not a consumer app: scope is the knee angle only, and videos are
replayed from file so that every model level sees exactly the same frames.

## Screenshots

<!-- SCREENSHOT: measurement view with video, detected keypoints and knee angle -->
<!-- SCREENSHOT: full sequence run FP32 → FP16 → INT8 with session export -->
<!-- SCREENSHOT: one of the evaluation plots (e.g. Bland-Altman) -->

## Stack

React 18 · TypeScript · Vite · Tailwind CSS · TensorFlow.js (TFLite, WASM) ·
Python (NumPy, SciPy, pandas, Matplotlib) for the statistical evaluation

## Getting started

```bash
npm install                 # .npmrc sets legacy-peer-deps=true
# place the three .tflite models in public/models/
npm run measure             # build + preview on http://localhost:4173

pip install -r scripts/requirements.txt
python3 scripts/test_postprocess_benchmark.py
```

Measurements only run against `npm run measure`, never against the dev server, because
HMR and source maps would skew the latency. Architecture, inference pipeline,
measurement workflow and the Python postprocessing are documented in
[docs/measurement.md](docs/measurement.md).

## What mattered to me

- **A fair comparison.** Deterministic replay seeks to the midpoint of each frame and
  waits for the `seeked` event, so FP32, FP16 and INT8 process exactly the same frames.
  A frame whose seek times out is recorded as dropped instead of silently replaced.
- **Reproducibility.** Runtime dependencies are pinned to exact versions, and every
  exported session records the model's SHA-256 fingerprint, the threading path actually
  used, cross-origin isolation and the TF.js versions.
- **Clean measurement.** Latency covers only `predict()` and the readback; pre- and
  post-processing are excluded. Multi-threaded WASM is detected at runtime, with a
  fallback to a single thread.
- **Statistics outside the app.** The app only records; MAE/RMSE, Bland-Altman and the
  latency tests (Friedman, Holm-corrected Wilcoxon) run in a tested Python package.
