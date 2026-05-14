# EdgeFit Pro — KI-Übersicht

Bachelorarbeit-POC. React + TS + Vite + Tailwind. Browser-only Bike-Fitting via TensorFlow.js **TFLite** MoveNet (Lightning, WASM-Backend) mit **3 Quantisierungsstufen (fp32 / fp16 / int8)** für Performance-Benchmarks. 100% lokal, kein Backend.

## Stack
- React 18 + TypeScript 5 + Vite 6 + TailwindCSS 3
- `@tensorflow/tfjs-tflite` (^0.0.1-alpha.10) — TFLite Modell-Loader
- `@tensorflow/tfjs-backend-wasm` (^4.22.0) — Inferenz-Backend
- `@tensorflow/tfjs-core` / `tfjs-converter` (^4.22.0)
- `@tensorflow-models/pose-detection` (^2.1.3) — nur noch als Typ-Quelle (`Pose`, `Keypoint`)
- Persistenz: `localStorage` (key `edgefit_history`, max 20 Einträge) + Benchmark-JSON-Download

> Hinweis: `npm install` benötigt `--legacy-peer-deps`, da `tfjs-tflite@0.0.1-alpha.x` einen veralteten Peer auf `tfjs-core@4.9.0` deklariert.

## Scripts
- `npm run dev` — Vite Dev-Server
- `npm run build` — `tsc -b && vite build`
- `npm run preview`

## Vite-Konfiguration (`vite.config.ts`)
- `optimizeDeps.include: ['@tensorflow/tfjs-tflite']` — esbuild-Prebundle erzwingen.
- `build.rollupOptions.external: [/tflite_web_api_client/]` — internes Lazy-Modul von tfjs-tflite, das Rollup nicht statisch auflösen kann. Wird zur Laufzeit aus `node_modules/@tensorflow/tfjs-tflite/dist/` gezogen.

## Datenfluss
```
Webcam → <video> → usePoseDetection (TFLite + WASM) → Pose (17 keypoints)
  ├─→ VideoCanvas (Skeleton overlay, Live-Winkel)         [home view]
  └─→ AnalysisView (sammelt 5 Pedalzyklen)
        ├─→ BiomechanicalAnalyzer → AnalysisResults
        │     → RecommendationsEngine → RecommendationReport
        │     → ResultsDashboard + HistoryStorage → localStorage
        │     → HistoryView
        └─→ onFrameMeasurement(pose)
              → App.handleFrameMeasurement
              → benchmarkExporter.recordFrame(...)
              → JSON-Export (Browser-Download)
```

## Datei-Map (`src/`)

### Code
| Datei | Rolle |
|---|---|
| `main.tsx` | React-Mount |
| `App.tsx` | View-Router (`home\|analysis\|results\|history`), Hook-Konsument, Benchmark-Verdrahtung, Level-Wechsel, Frame-Recording |
| `hooks/usePoseDetection.ts` | **Neu:** TFLite-Loader (`loadModel(level)`), WASM-Backend, `detectPose(video) → Pose\|null`, Warmup-Tracking (30 Frames), `lastMeasurementRef` für synchronen Frame-Snapshot, sauberes Tensor-Cleanup |
| `components/VideoCanvas.tsx` | Webcam + Canvas-Overlay (Live-Preview, home view) |
| `components/AnalysisView.tsx` | Erfassungs-UI (5 Zyklen), ruft `detectPose` pro Frame, leitet via `onFrameMeasurement(pose)` weiter |
| `components/QuantizationControls.tsx` | **Neu:** FP32/FP16/INT8-Buttons, Teilnehmer-ID-Input, Warmup-Badge, JSON-Export-Button |
| `components/MetricsOverlay.tsx` | Live-Metriken Sidebar (FPS, Inferenzzeit) |
| `components/ResultsDashboard.tsx` | Ergebnis-Visualisierung + Empfehlungen |
| `components/HistoryView.tsx` | Verlaufsliste |
| `services/BiomechanicalAnalyzer.ts` | Pedalzyklus-Detektor (FSM `searching→extension→flexion`), Statistik, Singleton |
| `services/RecommendationsEngine.ts` | `generateRecommendations(results)` → priorisierte Empfehlungen |
| `services/HistoryStorage.ts` | `saveToHistory`, `loadHistory` (rawCycles werden NICHT gespeichert) |
| `services/BenchmarkExporter.ts` | **Neu:** Singleton, sammelt `FrameMeasurement[]`, exportiert `BenchmarkSession` als JSON-Download |
| `utils/AngleCalculator.ts` | Vektor-Winkel, `OPTIMAL_RANGES`, `evaluateAngle` |
| `types/quantization.ts` | **Neu:** `QuantizationLevel`, `TFLITE_MODEL_URLS`, `KP`-Indizes, `KEYPOINT_NAMES`, `WARMUP_FRAMES=30`, `MODEL_INPUT_SIZE=192` |

## Zentrale Typen
```ts
QuantizationLevel = 'fp32' | 'fp16' | 'int8'
Pose = poseDetection.Pose          // { keypoints: Keypoint[17] }
LastMeasurement { inferenceMs, fps, frameIndex, isWarmup }

FrameMeasurement {
  frameIndex, timestampMs, inferenceMs, fps,
  kneeAngleRight: number|null, kneeAngleLeft: number|null,
  keypointScores: number[17], isWarmup: boolean
}
BenchmarkSession {
  participantId, quantizationLevel, startTimestamp,
  systemInfo { userAgent, hardwareConcurrency, deviceMemory? },
  warmupFrames, frames: FrameMeasurement[]
}
BenchmarkSummary {
  totalFrames, validFrames,
  meanInferenceMs, stdInferenceMs, p50Ms, p95Ms,
  meanFps, meanKneeAngleRight, meanKneeAngleLeft
}
```

Domänen-Typen unverändert: `BiomechanicalAngles`, `AnalysisResults`, `AngleStatus`, `PedalCycle`, `Recommendation`.

## Inferenz-Pipeline (`usePoseDetection.detectPose`)
1. `tf.browser.fromPixels(video)` → `tf.image.resizeBilinear → [192,192,3]`
2. INT8: `tf.cast(..., 'int32')` (uint8-Input erwartet) — FP32/FP16: float32
3. `tf.expandDims(..., 0)` → `[1,192,192,3]`
4. `model.predict(input)` → `await output.data()` (synchronisiert Backend)
5. Output `[1,1,17,3]` flat parsen: `y,x,score` je Keypoint → in Pixel skalieren
6. Tensoren disposen (input + output) — Input-Pipeline via `tf.tidy`

## Optimale Winkelbereiche (`AngleCalculator.ts`)
- knee gestreckt: 140–150° • knee gebeugt: 65–75° (`KNEE_FLEXION_RANGE`)
- hip: 40–50° • ankle: 90–110° • elbow: 150–170° • back: 40–50°
- `evaluateAngle(angle, range, tol=5)` → optimal/acceptable/critical

## MoveNet Keypoints
`0 nose, 5/6 shoulder, 7/8 elbow, 9/10 wrist, 11/12 hip, 13/14 knee, 15/16 ankle` (L/R).
Konstante `KP` in `types/quantization.ts` exportiert Hip/Knee/Ankle-Indizes.

## Zyklus-Erkennung (Analyzer-FSM)
- Glättung: Moving Average über `smoothingWindow=5`
- Schwellen: `kneeThresholdHigh=120°` / `kneeThresholdLow=100°`
- Min Frames/Zyklus: 10 • Min Confidence: 0.5 • `targetCycles=5`

## Benchmark-Workflow
1. App startet → Default `fp32`-Modell wird geladen → Session `P01` automatisch gestartet.
2. In QuantizationControls Teilnehmer-ID setzen / Stufe wählen (Level-Wechsel ruft `loadModel(level)` + startet neue Session).
3. 'Analyse starten' → 5 Pedalzyklen werden aufgezeichnet, jeder Frame nach Warmup landet in `benchmarkExporter`.
4. 'JSON exportieren' → Download `benchmark_${participantId}_${level}_${ISO-Timestamp}.json`.
5. Python-Postprocessing: `pd.DataFrame(session["frames"])`.

## Erweiterungs-Hotspots
- **Neuer Winkel:** Funktion in `AngleCalculator.ts` + `OPTIMAL_RANGES`-Eintrag + Feld in `BiomechanicalAngles` + Erfassung in `AnalysisView` + Statistik in `BiomechanicalAnalyzer.getResults` + Eval-Regel in `RecommendationsEngine`.
- **Neue Empfehlung:** `analyzeXxx`-Funktion in `RecommendationsEngine.ts`, in `generateRecommendations` aufrufen.
- **Modell-URLs ändern:** `TFLITE_MODEL_URLS` in `types/quantization.ts` (z.B. lokaler Mirror falls TFHub blockiert).
- **Anderes Modell / Backend:** `usePoseDetection.ts` → `loadTFLiteModel`-Aufruf bzw. `tf.setBackend('wasm'|'webgl'|...)`. Output-Parsing in `detectPose` ggf. anpassen (`[1,1,17,3]`-Annahme).
- **Andere Persistenz:** `HistoryStorage.ts` (Analyse) bzw. `BenchmarkExporter.ts` (Benchmarks) — beide LocalStorage-/Download-only.
- **Schwellen/Targets tunen:** `DEFAULT_CONFIG` in `BiomechanicalAnalyzer.ts`, `WARMUP_FRAMES` in `types/quantization.ts`.
- **Zusätzliche Benchmark-Metriken:** `FrameMeasurement` erweitern + `recordFrame`-Aufruf in `App.handleFrameMeasurement` ergänzen.

## Konventionen / Gotchas
- UI/Kommentare auf Deutsch.
- Keine Tests, kein Backend, keine ENV-Vars.
- Singleton-Pattern für Analyzer (`getAnalyzer`/`resetAnalyzer`) und Exporter (`benchmarkExporter`).
- Canvas-Y zeigt nach unten (relevant für `calculateBackAngle`).
- `rawCycles` werden bewusst NICHT in History serialisiert (zu groß).
- TFLite Backend (`wasm`) muss vor `loadTFLiteModel` initialisiert sein (`setBackend('wasm')` + `tf.ready()` — passiert einmalig im Hook via `backendReadyRef`).
- `lastMeasurementRef` ist ein React-Ref, das `detectPose` synchron befüllt — App.tsx kann nach `await detectPose(...)` ohne setState-Race darauf zugreifen.
- INT8-Modell erwartet uint8-Input → in der Pipeline wird das `resizeBilinear`-Float-Resultat per `tf.cast(_, 'int32')` zurück gewandelt.
- `--legacy-peer-deps` erforderlich beim Installieren neuer Dependencies.
