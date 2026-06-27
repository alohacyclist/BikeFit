# EdgeFit Pro

Bachelorarbeit-Mess-App. Browser-only Kniewinkel-Messung via TensorFlow.js **TFLite** MoveNet (SinglePose Lightning, WASM-Backend) mit **drei Quantisierungsstufen (fp32 / fp16 / int8)**. Genauigkeitsvalidierung gegen externe 2D-Video-Analyse-Software (Ground Truth).

**Forschungsfrage:** Wie verändert die Post-Training-Quantisierung des MoveNet-Lightning-Modells die per-Frame-Genauigkeit der Kniewinkel-Schätzung und die Inferenzlatenz?

**Scope:** ausschließlich Kniewinkel, ausschließlich Datei-Replay (deterministische Frame-Sequenz, paired comparison über alle Stufen).

## Stack

- React 18 + TypeScript 5 + Vite 6 + TailwindCSS 3
- `@tensorflow/tfjs-tflite` **0.0.1-alpha.9** (exakt gepinnt)
- `@tensorflow/tfjs-backend-wasm` 4.22.0 + `tfjs-core` 4.22.0
- `@tensorflow-models/pose-detection` 2.1.3 (nur Typen)

Alle Versionen gepinnt (kein Caret) für Reproduzierbarkeit. `.npmrc` setzt `legacy-peer-deps=true` wegen TFLite-Alpha-Peers.

## Scripts

- `npm run dev` — Vite Dev-Server (**nicht für Messungen**)
- `npm run build` — Production Build
- `npm run preview` — serviert `dist/` Build
- `npm run measure` — `build` + `preview --host` (Mess-Modus)

## Mess-Modus — wichtig

Messungen ausschließlich gegen `npm run measure`, **niemals gegen `npm run dev`**. Dev-Modus läuft mit HMR, Source-Maps und WebSocket-Watcher — verzerrt die Inferenzlatenz-Messung.

## Vite-Konfiguration

`vite.config.ts` setzt auf dev + preview server:
```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```
Damit ist `SharedArrayBuffer` verfügbar und WASM-Multi-Threading aktivierbar. `optimizeDeps.exclude: ['@tensorflow/tfjs-tflite']`, da das Paket UMD-only ist (Loading via dynamic script tag im Hook).

## Architektur — Datenfluss

```
CFR-konvertierte .mp4
  → useVideoSource bindet Datei an <video>
  → AnalysisView.recordingLoop:
       for i in 0..totalFrames-1:
         seekTo((i+0.5)/targetFps)        # frame-genau
         detectPose(video)                # TFLite-Inferenz
         calculateKneeAngle(hip,knee,ankle) für lockedSide
         analyzer.addFrame(...) + benchmarkExporter.recordFrame(...)
  → JSON-Download → externes Python-Postprocessing
```

## Datei-Map (`src/`)

| Datei | Rolle |
|---|---|
| `main.tsx` | React-Mount |
| `App.tsx` | Single-View-Layout, State-Orchestrierung, Wiring aller Hooks |
| `hooks/usePoseDetection.ts` | TFLite-Loader, `loadModel`/`resetBackend`/`beginWarmup`, Multi-Threading mit Fail-Safe, SHA-256-Fingerprint, per-Frame `detectPose(video, {record?})` |
| `hooks/useVideoSource.ts` | Bindet Replay-Datei an `<video>`, berechnet `totalFrames = floor(duration × targetFps)` |
| `components/AnalysisView.tsx` | Setup-Preview + deterministischer Recording-Loop (Seek-basiert), Skeleton-Render, hartes Abort bei Seek-Fehler |
| `components/VideoSourcePanel.tsx` | UI: Datei-Upload + Ziel-Framerate (24/25/30/60) |
| `components/QuantizationControls.tsx` | UI: Quant-Stufe, Side-Lock, Threading-Toggle, Probanden-ID, JSON-Export. Während Recording gesperrt. |
| `services/BiomechanicalAnalyzer.ts` | Zyklus-FSM auf Knie (Schwellen 135°/115°), Hold-Last-Value, Statistik |
| `services/BenchmarkExporter.ts` | Pro-Frame-Sammlung, `BenchmarkSession`-Schema, Validierungs-Metriken, JSON-Download |
| `utils/AngleCalculator.ts` | Vektor-Winkel, `calculateKneeAngle`, `SIDE_KEYPOINTS`, `BodySide`, `isKneeTripleValid` |
| `utils/modelFingerprint.ts` | SHA-256-Hash + Größe + URL via WebCrypto |
| `types/quantization.ts` | `QuantizationLevel`, lokale Modell-Pfade, MoveNet-Konstanten |

## Zentrale Typen

```ts
QuantizationLevel = 'fp32' | 'fp16' | 'int8'
BodySide = 'left' | 'right'
ThreadingPreference = 'single' | 'multi'

FrameMeasurement {
  frameIndex, timestampMs, inferenceMs, fps,
  kneeAngleRight: number|null, kneeAngleLeft: number|null,
  keypointScores: number[17], isWarmup: boolean
}

ModelFingerprint { url, sha256, sizeBytes, loadedAt }

BenchmarkSession {
  participantId, quantizationLevel, startTimestamp,
  systemInfo: { userAgent, hardwareConcurrency, deviceMemory,
                crossOriginIsolated, sharedArrayBufferAvailable },
  warmupFrames, lockedSide,
  modelFingerprint, threadingMode, numThreads,
  videoSource: 'file', videoSourceName, targetFps,
  durationSeconds, expectedFrames,
  frames: FrameMeasurement[]
}
```

App rechnet keine Aggregate. Selektion gültiger Frames (anhand `keypointScores`)
+ jede statistische Auswertung erfolgt im Python-Postprocessing mit dokumentierter,
variierbarer Confidence-Schwelle.

## Inferenz-Pipeline

1. `tf.browser.fromPixels(video)` → `[H,W,3]`
2. `tf.image.resizeBilinear → [192,192,3]`
3. INT8: `tf.cast(_, 'int32')`; FP32/FP16: float32
4. `tf.expandDims(_, 0)` → `[1,192,192,3]`
5. `model.predict(input)` → `await output.data()` (WASM-Sync)
6. Output `[1,1,17,3]` flat parsen → 17×{y,x,score} → Bildkoordinaten
7. Tensoren disposen (input + output)

**Latenz-Messung** umschließt nur `predict()` + `await data()`. Pre-/Post-Processing ist modell-unabhängig und nicht Teil der Messung.

## Deterministisches Replay-Stepping

Zentraler Architekturpunkt für die wissenschaftliche Validität:

```ts
for (let i = 0; i < total; i++) {
  const ok = await seekTo(video, (i + 0.5) / targetFps);
  if (!ok) abort();                       // Hartes Abort, kein stilles Falsch-Frame
  const pose = await detectPose(video);
  // ...
}
```

- `seekTo` wartet auf das `seeked`-Event (5 s Timeout als Safety-Net).
- Frame-Mitte `(i+0.5)/fps` ist robust gegen Frame-Boundary-Rundung.
- Inferenz-Tempo entkoppelt vom Frame-Set: schnelle Modelle liefern nicht mehr Frames als langsame.
- **Garantie:** FP32, FP16 und INT8 verarbeiten exakt dieselben `totalFrames = floor(duration × targetFps)` Frames.

## Zyklus-Erkennung

FSM über geglätteten Kniewinkel (Moving Average, Window=5):
```
searching → extension (knee > 135°)
extension → flexion  (knee < 115°)
flexion → extension  (knee > 135°) → Zyklus++
```
Hold-Last-Value bei kurzen Detection-Aussetzern (max. 5 Frames). Zykluszahl ist Ergebnis, kein Stop-Kriterium — der Loop läuft immer durch alle Video-Frames.

## Multi-Threading

Multi-Thread-WASM aktiviert sich automatisch, wenn:
1. COOP/COEP-Header in Vite-Config greifen (gesetzt).
2. Browser meldet `self.crossOriginIsolated === true`.
3. `SharedArrayBuffer` ist im Kontext definiert.

Bei erfüllten Bedingungen wird `numThreads = min(hardwareConcurrency, 4)` an `loadTFLiteModel` durchgereicht. Andernfalls Fail-Safe-Fallback auf Single-Thread, ehrlich im Export geloggt (`threadingMode`, `numThreads`).

**Verifikation in DevTools:**
```js
self.crossOriginIsolated     // muss true sein
typeof SharedArrayBuffer     // "function"
```

## Wissenschaftlicher Mess-Workflow

```
1. Video aufzeichnen (externe Kamera/Webcam, beliebiges Tool)
2. CFR-Konversion:
   ffmpeg -i in.mov -vf fps=30 -fps_mode cfr -c:v libx264 \
          -g 1 -keyint_min 1 -pix_fmt yuv420p replay.mp4
3. Ground Truth: replay.mp4 in 2D-Software (Kinovea o.ä.)
   → per-Frame Kniewinkel → GT.csv (frameIndex, timestamp, kneeAngle)
4. App: npm run measure → http://localhost:4173
5. Datei replay.mp4 laden, targetFps=30 setzen
6. Seite wählen (muss mit GT-Annotation übereinstimmen)
7. Threading wählen (Single/Multi)
8. Quantisierung FP32 → "Analyse starten" → JSON-Export
9. Quantisierung wechseln (Backend wird auto-resettet) → erneut → Export
10. Wiederholen für INT8
11. Python: paired Frame-Vergleich (frameIndex join) der 3 JSONs vs GT.csv
    → MAE/RMSE/Bias-Plot/Bland-Altman pro Stufe
    → Friedman + Holm-Wilcoxon für Latenz
```

## Modell-Pfade

`public/models/`:
- `movenet-lightning-fp32.tflite` (~9.3 MB)
- `movenet-lightning-fp16.tflite` (~4.7 MB)
- `movenet-lightning-int8.tflite` (~2.8 MB)

Download von [Kaggle MoveNet TFLite](https://www.kaggle.com/models/google/movenet/tfLite/singlepose-lightning) — siehe `public/models/README.md`. Der SHA-256-Hash der geladenen Datei wird pro Session im Export protokolliert.

## Verifikations-Mechanismen pro Session

Im exportierten JSON:

| Feld | Bedeutung |
|---|---|
| `modelFingerprint.sha256` | Modell-Identität — bei Mismatch zwischen Sessions ist die Messung nicht vergleichbar |
| `threadingMode` + `numThreads` | tatsächlich genutzter Threading-Pfad (nicht selbst-gesetzte Flag) |
| `systemInfo.crossOriginIsolated` | Browser-bestätigte COI — ohne diese kein echtes MT möglich |
| `targetFps` | CFR-Framerate für deterministisches Stepping |
| `lockedSide` | fix gewählte Körperseite (muss mit GT übereinstimmen) |
| `frames[i].keypointScores[17]` | Roh-Confidence pro Keypoint — Selektion gültiger Frames erfolgt in Python (variierbare Schwelle, Sensitivity-Sweep) |

## Konventionen / Gotchas

- UI/Kommentare auf Deutsch.
- Per-Frame `isWarmup: true` für die ersten 30 realen Inferenzen — Python filtert.
- Bei Seek-Fehler im Recording-Loop: **harter Abort**, Session wird verworfen. Keine Teil-Daten im Export.
- Während Recording sind Quantisierung, Side und Threading im UI gesperrt.
- File-Replay erwartet **CFR-Video** — VFR-Aufnahmen erst ffmpeg-konvertieren.
- Backend wird beim Stufen- oder Threading-Wechsel komplett zurückgesetzt (`tf.engine().reset()`), damit kein JIT/Cache-Carryover die Latenz verfälscht.
- Modell-Fingerprint pro Session geloggt → Re-Run-Reproduzierbarkeit verifizierbar.
- GitHub Pages als Deploy-Target funktioniert nicht für Multi-Threading (kein Custom-Header-Support).
