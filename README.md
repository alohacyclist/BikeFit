# EdgeFit Pro — KI-Übersicht

Bachelorarbeit-POC. React + TS + Vite + Tailwind. Browser-only Kniewinkel-Messung via TensorFlow.js **TFLite** MoveNet (Lightning, WASM-Backend) mit **3 Quantisierungsstufen (fp32 / fp16 / int8)** für Performance-Benchmarks. Ground-Truth-Validierung gegen externe 2D-Video-Analyse-Software.

**Scope:** ausschließlich Kniewinkel (Hüfte / Knöchel / Ellbogen / Rücken bewusst entfernt). Studienleiter-getriebene Datensammlung, Auswertung extern in Python.

## Stack
- React 18 + TypeScript 5 + Vite 6 + TailwindCSS 3
- `@tensorflow/tfjs-tflite` **0.0.1-alpha.9** (exakt gepinnt — alpha.10 fehlt WASM-Binaries)
- `@tensorflow/tfjs-backend-wasm` 4.22.0 + `tfjs-core` 4.22.0
- `@tensorflow-models/pose-detection` 2.1.3 (nur Typen)
- Persistenz: JSON-Download pro Session

## Scripts
- `npm run dev` — Vite Dev-Server (**nicht für Messungen — siehe unten**)
- `npm run build` — Production Build
- `npm run preview` — serviert `dist/` Build (Mess-Modus)
- `npm run measure` — `build` + `preview` in einem Schritt mit `--host`

## Messmodus — wichtig

Messungen ausschließlich gegen `npm run measure` (= `vite build && vite preview`) ausführen, **niemals gegen `npm run dev`**. Im Dev-Modus laufen HMR, Source-Maps und ein WebSocket-Watcher, die GC-Pausen verursachen und die Inferenzlatenz-Messung um zweistellige Prozentpunkte verzerren können.

## Vite-Konfiguration
- `optimizeDeps.exclude: ['@tensorflow/tfjs-tflite']` — Paket UMD-only.
- TFLite-UMD wird zur Laufzeit dynamisch geladen (nach `window.tf = tf`).

## Architektur — Datenfluss
```
Video-Quelle (Webcam | Datei-Replay)
  → <video> Element (gesteuert via useVideoSource)
  → detectPose(video) → Pose (17 keypoints)
  → AnalysisView → BiomechanicalAnalyzer (Pedalzyklus-FSM)
  → onFrameMeasurement → benchmarkExporter.recordFrame
  → JSON-Export (Browser-Download) → Python-Postprocessing
```

## Datei-Map (`src/`)

| Datei | Rolle |
|---|---|
| `main.tsx` | React-Mount |
| `App.tsx` | Single-View-Layout, hält Settings, verdrahtet alle Hooks |
| `hooks/usePoseDetection.ts` | TFLite-Loader, `loadModel` / `resetBackend`, SHA-256-Fingerprint, Threading-Setting, Warmup, `detectPose(video) → Pose` |
| `hooks/useVideoSource.ts` | Abstraktion Webcam ↔ Datei für `<video>`-Element |
| `components/AnalysisView.tsx` | Recording-Loop, Side-Lock, Live-Skeleton, Knie-Winkel-Overlay |
| `components/QuantizationControls.tsx` | UI: Quantisierung, Side-Lock, Threading-Toggle, Probanden-ID, JSON-Export |
| `components/VideoSourcePanel.tsx` | UI: Webcam-vs-Datei + integrierte Aufnahme zu .webm-Download |
| `services/BiomechanicalAnalyzer.ts` | Zyklus-FSM auf Knie, deskriptive Statistik (Knie-Extension/Flexion) |
| `services/BenchmarkExporter.ts` | Pro-Frame-Sammlung + Validierungsmetriken + Modell-Fingerprint + Threading-Mode + Video-Quelle |
| `services/VideoRecorder.ts` | MediaRecorder-Wrapper für Referenz-Video-Aufnahme |
| `utils/AngleCalculator.ts` | Vektor-Winkel, `calculateKneeAngle`, `isKneeTripleValid`, `SIDE_KEYPOINTS`, `BodySide` |
| `utils/modelFingerprint.ts` | SHA-256-Hash + Größe + URL des `.tflite`-Files |
| `types/quantization.ts` | `QuantizationLevel`, lokale Modell-Pfade, `KP`-Indizes, `WARMUP_FRAMES=30` |

## Zentrale Typen
```ts
QuantizationLevel = 'fp32' | 'fp16' | 'int8'
BodySide = 'left' | 'right'
ThreadingPreference = 'single' | 'multi'
Pose = { keypoints: Keypoint[17] }
BiomechanicalAngles = { knee: number | null }

FrameMeasurement {
  frameIndex, timestampMs, inferenceMs, fps,
  kneeAngleRight: number|null, kneeAngleLeft: number|null,
  keypointScores: number[17], isWarmup: boolean
}
ModelFingerprint { url, sha256, sizeBytes, loadedAt }
BenchmarkSession {
  participantId, quantizationLevel, startTimestamp,
  systemInfo, warmupFrames, lockedSide,
  modelFingerprint, threadingMode, videoSource, videoSourceName?,
  frames: FrameMeasurement[],
  validationMetrics?: { validKneeRatio*, meanKeypointScores, interpolatedFrames, totalCyclesDetected }
}
```

## Inferenz-Pipeline (`usePoseDetection.detectPose`)
1. `tf.browser.fromPixels(video)` → `[H,W,3]`
2. `tf.image.resizeBilinear → [192,192,3]`
3. INT8: `tf.cast(_, 'int32')` (uint8-Input); FP32/FP16: float32
4. `tf.expandDims(_, 0)` → `[1,192,192,3]`
5. `model.predict(input)` → `await output.data()` (Backend-Sync)
6. Output `[1,1,17,3]` flat parsen → 17×{y,x,score} → in Bildkoordinaten skalieren
7. Tensoren disposen (input + output)

## Zyklus-Erkennung
FSM über geglätteten Kniewinkel:
```
searching → extension (knee > 135°)
extension → flexion  (knee < 115°)
flexion → extension  (knee > 135°) → Zyklus++
```
Hold-Last-Value-Fallback bei kurzen Detection-Aussetzern (max. 5 Frames).

## Wissenschaftlicher Mess-Workflow

```
1. Aufnahme: Webcam-Modus → "● Aufnahme starten" → "■ Stop & Download" → record.webm
2. CFR-Konversion: ffmpeg -i record.webm -c:v libx264 -r 30 -vsync cfr -g 1 -pix_fmt yuv420p replay.mp4
3. Ground Truth: replay.mp4 in 2D-Software (Kinovea o.ä.) → per-Frame Kniewinkel → GT.csv
4. App: Replay-Datei-Modus + replay.mp4 laden + Quantisierung FP32 → Replay starten → JSON-Export
5. Threading + Quantisierung wechseln → Replay erneut → Export FP16
6. dito INT8
7. Python: paired Frame-Vergleich (frameIndex Join) der drei JSONs gegen GT.csv
```

Quantisierungsstufen-Wechsel triggert automatisch Backend-Reset (`tf.engine().reset()`), damit JIT/Cache-Carryover die Latenz nicht verfälscht.

## Modell-Pfade
- `public/models/movenet-lightning-fp32.tflite` (~9.3 MB)
- `public/models/movenet-lightning-fp16.tflite` (~4.7 MB)
- `public/models/movenet-lightning-int8.tflite` (~2.8 MB)

Download von [Kaggle MoveNet TFLite](https://www.kaggle.com/models/google/movenet/tfLite/singlepose-lightning) — siehe `public/models/README.md`.

## WASM-Threading-Hinweis
Multi-Threading benötigt `SharedArrayBuffer`, das nur mit COOP/COEP-Headern verfügbar ist:
```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```
Diese sind aktuell nicht gesetzt → effektiver Modus ist **single-thread**. Die UI zeigt den aktiven Modus an und loggt ihn im JSON-Export. Für Multi-Thread-Vergleich: Vite-Plugin oder Reverse-Proxy konfigurieren + WASM-Binaries lokal hosten.

## Konventionen / Gotchas
- UI/Kommentare auf Deutsch.
- `--legacy-peer-deps` (via `.npmrc`) wegen tfjs-tflite alpha-9 peer.
- Singleton-Pattern für `benchmarkExporter`.
- File-Replay erwartet **CFR-Video** — VFR-Aufnahmen erst ffmpeg-konvertieren.
- Mess-Sessions: ein Quantisierungs-Wechsel pro Session, dazwischen `resetBackend` (passiert automatisch im UI-Flow).
- Modell-Fingerprint pro Session geloggt → Re-Run-Reproduzierbarkeit verifizierbar.
