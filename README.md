# EdgeFit Pro

Bachelor-thesis measurement app. Browser-only knee-angle measurement via TensorFlow.js **TFLite** MoveNet (SinglePose Lightning, WASM backend) across **three post-training quantization levels (fp32 / fp16 / int8)**. Accuracy is validated per frame against an external 2D video-analysis tool (ground truth).

**Research questions:**

- **FF1** — How accurate is the *un-quantized* FP32 baseline against the tracker reference (MAE, RMSE)?
- **FF2** — How does post-training quantization change per-frame knee-angle accuracy (MAE/RMSE per level, pairwise Δθ vs FP32, Bland-Altman)?
- **FF3** — How does quantization (× threading) change inference latency (mean, SD, median, P95; Friedman + Holm-Wilcoxon)?

**Scope:** knee angle only, file-replay only (deterministic frame sequence, paired comparison across all levels).

## Stack

- React 18 + TypeScript 5 + Vite 6 + TailwindCSS 3
- `@tensorflow/tfjs-tflite` **0.0.1-alpha.9** (exact pin)
- `@tensorflow/tfjs-core` / `-backend-wasm` / `-backend-webgl` / `-converter` 4.22.0
- `@tensorflow-models/pose-detection` 2.1.3 (types only)

All versions are pinned (no caret) for reproducibility. `.npmrc` sets `legacy-peer-deps=true` because of the TFLite-alpha peer ranges.

## Scripts

- `npm run dev` — Vite dev server (**not for measurements**)
- `npm run build` — production build (`tsc -b && vite build`)
- `npm run preview` — serves the `dist/` build
- `npm run measure` — `build` + `preview --host` (measurement mode)

## Measurement mode — important

Measure only against `npm run measure`, **never against `npm run dev`**. Dev mode runs HMR, source maps and a WebSocket watcher — all of which skew the inference-latency measurement.

## Vite configuration

`vite.config.ts` sets on both the dev and preview server:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

This makes `SharedArrayBuffer` available and enables WASM multi-threading. `optimizeDeps.exclude: ['@tensorflow/tfjs-tflite']` because the package is UMD-only (loaded via a dynamic script tag in the hook).

## Architecture — data flow

```
CFR-converted .mp4 (cached in IndexedDB across reloads)
  → useVideoSource binds the file to <video>
  → AnalysisView recording loop:
       for i in 0..videoTotalFrames-1:
         seekTo((i+0.5)/targetFps)          # frame-accurate
         detectPose(video, {record})        # TFLite inference
         → BenchmarkExporter.recordFrame(...)  # raw per-frame data only
  → BenchmarkSession JSON (schema v1.1.0)
  → downloaded, then handed to the Python postprocessing package
```

The app computes **no aggregates**. Selection of valid frames (via `keypointScores`) and every statistical evaluation happen in the Python postprocessing with a documented, tunable confidence threshold.

## Stage switching & full sequence

The core operational point of this build: **each quantization/threading change is a full `window.location.reload()`, not an in-process backend reset.** Hot-swapping the TFLite pthread pool deadlocks (qu8-delegate swap), so a fresh page boot guarantees a clean pthread pool and no JIT/cache carryover into the latency numbers.

To survive the reload:

- **Replay video** is cached in IndexedDB (`videoStore`) so it is not lost on boot.
- **Pending boot state** (level to load next, remaining levels, auto-record flag) is persisted in `sessionStorage` (`sequenceResume`).
- **Result JSONs** of a full sequence are collected in IndexedDB and downloaded **once at the end via a single user gesture** — a post-reload auto-download would be un-gestured and blocked by Chrome.

"Run full sequence" boots FP32 → records → reloads to FP16 → records → reloads to INT8 → records, then offers all three JSONs for download.

## File map (`src/`)

| File | Role |
|---|---|
| `main.tsx` | React mount |
| `App.tsx` | Single-view layout, state orchestration, full-sequence driver (reload-based), result collection + download |
| `hooks/usePoseDetection.ts` | TFLite loader (`loadModel`/`resetBackend`/`beginWarmup`), reload-based level/threading switch, multi-threading with fail-safe, SHA-256 fingerprint, per-frame `detectPose(video, {record?})` |
| `hooks/useVideoSource.ts` | Binds replay file to `<video>`, computes `totalFrames = floor(duration × targetFps)` |
| `components/AnalysisView.tsx` | Setup preview + deterministic seek-based recording loop, skeleton render, `droppedFrames` bookkeeping, hard abort past the dropped-frame ceiling |
| `components/VideoSourcePanel.tsx` | UI: file upload + target framerate (24/25/30/60) |
| `components/QuantizationControls.tsx` | UI: quant level, side lock, threading toggle, participant ID, single export / full-sequence run. Locked during recording. |
| `services/BenchmarkExporter.ts` | Pure per-frame collector, `BenchmarkSession` schema (v1.1.0), JSON download. No app-side filtering or statistics. |
| `utils/AngleCalculator.ts` | Vector angle, `calculateKneeAngle`, `SIDE_KEYPOINTS`, `BodySide`, `isKneeTripleValid` |
| `utils/modelFingerprint.ts` | SHA-256 hash + size + URL via WebCrypto |
| `utils/hardwareInfo.ts` | Best-effort device/CPU/OS/browser from UA; `toCompactTimestamp` for filenames |
| `utils/environment.ts` | v1.1.0 reproducibility block: tfjs versions, `userAgentData`, battery, COI, actual `numThreads`, manual power flags |
| `utils/videoStore.ts` | IndexedDB store for the replay file and the sequence result JSONs |
| `utils/sequenceResume.ts` | `sessionStorage` pending-boot state that survives the reload |
| `types/quantization.ts` | `QuantizationLevel`, local model paths, MoveNet constants (`WARMUP_FRAMES`, `MODEL_INPUT_SIZE`, keypoint indices/names) |

## Central types (schema v1.1.0)

```ts
QuantizationLevel = 'fp32' | 'fp16' | 'int8'
BodySide = 'left' | 'right'

FrameMeasurement {
  frameIndex, timestampMs, inferenceMs, fps,
  kneeAngleRight: number, kneeAngleLeft: number,
  keypoints: Point2D[17], keypointScores: number[17],
  isWarmup: boolean
}

DroppedFrame { frameIndex, reason, message? }
// reason: 'seek_timeout' | 'predict_error' | 'readback_error' | 'other'

BenchmarkSession {
  schemaVersion: '1.1.0',
  probandId, level, runIndex, createdAt,
  targetFps, bodySide, threading: 'single'|'multi',
  videoDurationSec, videoTotalFrames, warmupCount,
  modelFingerprintSha256, modelUrl, modelLoadMs,
  userAgent, hardware: HardwareInfo,
  environment?: EnvironmentInfo,        // optional; absent in legacy 1.0.0 data
  frames: FrameMeasurement[],
  droppedFrames: DroppedFrame[]
}
```

**Invariant:** `videoTotalFrames == frames.length + droppedFrames.length`.

## Inference pipeline

1. `tf.browser.fromPixels(video)` → `[H,W,3]`
2. `tf.image.resizeBilinear → [192,192,3]`
3. INT8: `tf.cast(_, 'int32')`; FP32/FP16: float32
4. `tf.expandDims(_, 0)` → `[1,192,192,3]`
5. `model.predict(input)` → `await output.data()` (WASM sync readback)
6. Output `[1,1,17,3]` parsed flat → 17×{y,x,score} → image coordinates
7. Dispose tensors (input + output)

**Latency measurement** wraps only `predict()` + `await data()`. Pre-/post-processing is model-independent and excluded from the measurement.

## Deterministic replay stepping

Central to the scientific validity of the comparison:

```ts
for (let i = 0; i < total; i++) {
  const ok = await seekTo(video, (i + 0.5) / targetFps);
  if (!ok) { markDropped(i, 'seek_timeout'); continue; }  // no silent wrong frame
  const pose = await detectPose(video);
  // ...
}
```

- `seekTo` waits for the `seeked` event (5 s timeout as a safety net).
- The frame midpoint `(i+0.5)/fps` is robust against frame-boundary rounding.
- Inference speed is decoupled from the frame set: fast models do not produce more frames than slow ones.
- **Guarantee:** FP32, FP16 and INT8 process exactly the same `videoTotalFrames = floor(duration × targetFps)` frames.

## Dropped frames & abort

A seek timeout or a `predict`/readback error records the frame in `droppedFrames[]` (with a reason) rather than substituting a wrong frame. The loop tolerates up to `MAX_DROPPED_FRAMES = 15`; past that it **hard-aborts** and the session is discarded (no partial data). This keeps the frame-count invariant honest while surviving occasional decoder hiccups.

## Multi-threading

Multi-thread WASM engages automatically when:

1. The COOP/COEP headers from the Vite config take effect (set).
2. The browser reports `self.crossOriginIsolated === true`.
3. `SharedArrayBuffer` is defined in the context.

When available and requested, `numThreads = min(hardwareConcurrency, 4)` is passed to `loadTFLiteModel` (capped at 4 — the tfjs MT speedup is sublinear and sync overhead dominates for a small model). If the threaded load fails, a fail-safe retries with `numThreads = 1`. The **actually used** path is logged in the export (`threading`, `environment.numThreads`, `activeThreadingMode`), not the requested flag.

Because the TFLite pthread pool cannot be hot-swapped, **toggling threading triggers a page reload** (persisted via `sessionStorage`) — same mechanism as the level switch.

**Verify in DevTools:**

```js
self.crossOriginIsolated     // must be true
typeof SharedArrayBuffer     // "function"
```

## Scientific measurement workflow

```
1. Record video (external camera/webcam, any tool)
2. CFR conversion:
   ffmpeg -i in.mov -vf fps=30 -fps_mode cfr -c:v libx264 \
          -g 1 -keyint_min 1 -pix_fmt yuv420p replay.mp4
3. Ground truth: replay.mp4 in a 2D tracker (Kinovea etc.)
   → per-frame knee angle → GT file (frameIndex, timestamp, kneeAngle)
4. App: npm run measure → http://localhost:4173
5. Load replay.mp4, set targetFps = 30
6. Choose side (must match the GT annotation)
7. Choose threading (single/multi)
8. "Run full sequence": FP32 → FP16 → INT8 (auto-record across reloads)
9. Download all three session JSONs at the end (single gesture)
10. Python: paired per-frame comparison (frameIndex join) of the 3 JSONs vs GT
    → FF1 baseline, FF2 quantization effect (MAE/RMSE/Bland-Altman),
      FF3 latency (Friedman + Holm-Wilcoxon)
```

## Model paths

`public/models/` (git-ignored, download manually):

- `movenet-lightning-fp32.tflite` (~9.3 MB / 8.9 MiB)
- `movenet-lightning-fp16.tflite` (~4.7 MB / 4.5 MiB)
- `movenet-lightning-int8.tflite` (~2.8 MB / 2.8 MiB)

Download from [Kaggle MoveNet TFLite](https://www.kaggle.com/models/google/movenet/tfLite/singlepose-lightning) — see [public/models/README.md](public/models/README.md). The SHA-256 hash of the loaded file is logged per session in the export.

## Python postprocessing

Package `scripts/postprocess/` (many small modules) with the CLI wrapper `scripts/postprocess_benchmark.py`. Consumes the three quantization JSONs of a full sequence plus the tracker ground truth and computes FF1/FF2/FF3. Requires `numpy`, `scipy`, `pandas`, `matplotlib`.

```bash
python3 scripts/postprocess_benchmark.py \
    --proband P01 \
    --gt data/gt/P01_tracker.txt --gt-format tracker-multi --gt-frame-offset -1 \
    --fp32 "exports/benchmark_P01_fp32_*.json" \
    --fp16 "exports/benchmark_P01_fp16_*.json" \
    --int8 "exports/benchmark_P01_int8_*.json" \
    --side left --warmup 5 --outlier-k 3.0 \
    --out results/P01/
```

Outputs (into `--out`): `summary.csv`, `summary_aggregated.csv`, `keypoint_quality.csv`, `environment.csv` (v1.1.0 sessions only), `cycle_summary.csv`, LaTeX tables (`table_ff1/ff2/ff3.tex`), plots (Bland-Altman, latency time series/distribution, angle/score time series, delta histograms), and `run_log.json`. Cycle detection and all statistics live here, not in the app.

Tests (stdlib `unittest`, no pytest needed):

```bash
python3 scripts/test_postprocess_benchmark.py
```

## Per-session verification mechanisms

In the exported JSON:

| Field | Meaning |
|---|---|
| `modelFingerprintSha256` | Model identity — a mismatch between sessions makes them non-comparable |
| `threading` + `environment.numThreads` | Actually used threading path (not a self-set flag) |
| `environment.crossOriginIsolated` | Browser-confirmed COI — without it no real MT is possible |
| `environment.tfjs.*` | Pinned tfjs/tflite versions used for the run |
| `targetFps` | CFR framerate for deterministic stepping |
| `bodySide` | Fixed body side (must match GT) |
| `frames[i].keypointScores[17]` | Raw per-keypoint confidence — valid-frame selection happens in Python (tunable threshold, sensitivity sweep) |
| `droppedFrames[]` | Frames skipped and why; frame-count invariant stays auditable |

## Conventions / gotchas

- UI text and code comments are in **German**.
- `isWarmup: true` on the first `WARMUP_FRAMES = 5` real inferences — Python filters them out.
- Individual bad frames go to `droppedFrames[]`; only exceeding `MAX_DROPPED_FRAMES = 15` hard-aborts and discards the session (no partial data).
- During recording, quantization, side and threading are locked in the UI.
- File replay expects a **CFR video** — convert VFR recordings with ffmpeg first.
- Every level/threading change is a full **page reload** (fresh pthread pool → no JIT/cache carryover, avoids the qu8-delegate deadlock). Replay video and result JSONs survive via IndexedDB.
- The model fingerprint is logged per session → re-run reproducibility is verifiable.
- GitHub Pages does **not** work as a deploy target for multi-threading (no custom-header support → no `crossOriginIsolated`).
