import { useState, useEffect, useRef, useCallback } from "react";
import * as poseDetection from "@tensorflow-models/pose-detection";
import * as tf from "@tensorflow/tfjs-core";
import { setWasmPaths } from "@tensorflow/tfjs-backend-wasm";

// tfjs-tflite wird als UMD-Script in index.html geladen (ESM-Import scheitert
// wegen fehlender Submodule). Stellt window.tflite bereit.
interface TFLiteModelLike {
  predict: (input: tf.Tensor | tf.Tensor[]) => tf.Tensor | tf.Tensor[];
  /** Nicht offiziell dokumentiert für alpha.9, aber vorhanden — versuchen wir. */
  cleanUp?: () => void;
  dispose?: () => void;
}
interface TFLiteLoadOptions {
  numThreads?: number;
}
interface TFLiteGlobal {
  loadTFLiteModel: (
    url: string,
    options?: TFLiteLoadOptions,
  ) => Promise<TFLiteModelLike>;
  setWasmPath?: (path: string) => void;
}
declare global {
  interface Window {
    tflite?: TFLiteGlobal;
  }
}

const TFLITE_UMD_URL =
  "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.9/dist/tf-tflite.min.js";

let tfliteScriptPromise: Promise<void> | null = null;

function loadTFLiteScript(): Promise<void> {
  if (window.tflite) return Promise.resolve();
  if (tfliteScriptPromise) return tfliteScriptPromise;
  tfliteScriptPromise = new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = TFLITE_UMD_URL;
    script.async = true;
    script.onload = () => resolve();
    script.onerror = () =>
      reject(new Error("TFLite UMD-Bundle konnte nicht geladen werden"));
    document.head.appendChild(script);
  });
  return tfliteScriptPromise;
}
import {
  QuantizationLevel,
  TFLITE_MODEL_URLS,
  WARMUP_FRAMES,
  MODEL_INPUT_SIZE,
  KEYPOINT_NAMES,
} from "../types/quantization";
import {
  fingerprintModel,
  type ModelFingerprint,
} from "../utils/modelFingerprint";
import type { DroppedFrameReason } from "../services/BenchmarkExporter";

export interface PoseMetrics {
  inferenceTime: number;
  fps: number;
  frameCount: number;
}

// Bleibt strukturkompatibel zu poseDetection.Pose (keypoints: {x,y,score,name})
export type Pose = poseDetection.Pose;

export interface LastMeasurement {
  inferenceMs: number;
  fps: number;
  frameIndex: number;
  /** performance.now() VOR predict — Vertrag-Feld FrameMeasurement.timestampMs. */
  timestampMs: number;
  isWarmup: boolean;
}

/**
 * Ergebnis eines detectPose-Aufrufs. Bei Erfolg ist `pose` gesetzt; bei
 * Misserfolg ist `pose` null und `reason` trägt den Vertrag-konformen Grund
 * (seek_timeout wird vom Aufrufer, nicht hier, vergeben).
 */
export interface DetectResult {
  pose: Pose | null;
  reason?: DroppedFrameReason;
  message?: string;
}

export type ThreadingPreference = "single" | "multi";

export interface UsePoseDetectionResult {
  detector: TFLiteModelLike | null;
  poses: Pose[];
  metrics: PoseMetrics;
  isLoading: boolean;
  error: string | null;
  detectPose: (
    video: HTMLVideoElement,
    opts?: { record?: boolean },
  ) => Promise<DetectResult>;
  loadModel: (level: QuantizationLevel) => Promise<void>;
  resetBackend: () => Promise<void>;
  /** Startet das Warmup-Fenster für die kommende Aufnahme neu. */
  beginWarmup: () => void;
  currentLevel: QuantizationLevel;
  isWarmingUp: boolean;
  lastMeasurementRef: React.MutableRefObject<LastMeasurement | null>;
  /**
   * modelLoadMs der laufenden Aufnahme: Zeit von loadModel-Start (Fetch) bis
   * zum ersten erfolgreichen NON-Warmup-predict. null bis dahin. beginWarmup
   * setzt zurück; erster valider Frame stempelt den Wert.
   */
  modelLoadMsRef: React.MutableRefObject<number | null>;
  /** Fingerprint des aktuell geladenen Modells (null vor erstem Load). */
  modelFingerprint: ModelFingerprint | null;
  /** Vom Nutzer gewählter Threading-Modus. */
  threadingPreference: ThreadingPreference;
  setThreadingPreference: (mode: ThreadingPreference) => void;
  /** Tatsächlich verfügbarer Threading-Status (SharedArrayBuffer + COOP/COEP). */
  multiThreadingAvailable: boolean;
  /** Effektiver Modus nach Backend-Init. */
  activeThreadingMode: "single" | "multi" | "unknown";
  /** Tatsächlich an TFLite übergebener numThreads-Wert (null vor erstem Load). */
  activeNumThreads: number | null;
}

/**
 * Custom Hook für MoveNet Pose Detection via TFLite + WASM-Backend.
 * Unterstützt drei Quantisierungsstufen (fp32 / fp16 / int8) für Benchmarks.
 */
export function usePoseDetection(
  initialLevel: QuantizationLevel = "fp32",
): UsePoseDetectionResult {
  const [detector, setDetector] = useState<TFLiteModelLike | null>(null);
  const [poses, setPoses] = useState<Pose[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentLevel, setCurrentLevel] =
    useState<QuantizationLevel>(initialLevel);
  const [isWarmingUp, setIsWarmingUp] = useState(true);
  const [modelFingerprint, setModelFingerprint] =
    useState<ModelFingerprint | null>(null);
  // Echte Multi-Threading-Verfügbarkeit: SharedArrayBuffer + Cross-Origin-Isolation.
  // Beides muss zur Laufzeit existieren — sonst startet die Pthread-Workerpool von
  // WASM erst gar nicht. crossOriginIsolated wird vom Browser nur gesetzt, wenn
  // die richtigen COOP/COEP-Header geliefert werden.
  const multiThreadingAvailable =
    typeof SharedArrayBuffer !== "undefined" &&
    typeof self !== "undefined" &&
    (self as unknown as { crossOriginIsolated?: boolean })
      .crossOriginIsolated === true;
  // tfjs-tflite UMD ist Page-Lifetime-Singleton: der WASM-Pthread-Pool wird
  // nur EINMAL pro Tab initialisiert. Threading-Wechsel zur Laufzeit deadlockt
  // (Atomics.wait auf belegten SAB-Slots). Daher: gewünschter Modus wird in
  // sessionStorage persistiert und beim Boot vor dem ersten loadModel gelesen;
  // Wechsel triggert window.location.reload().
  const persistedThreading =
    typeof sessionStorage !== "undefined"
      ? (sessionStorage.getItem(
          "edgefit.threading",
        ) as ThreadingPreference | null)
      : null;
  const initialThreading: ThreadingPreference =
    persistedThreading === "single" || persistedThreading === "multi"
      ? persistedThreading
      : multiThreadingAvailable
        ? "multi"
        : "single";
  const [threadingPreference, setThreadingPreferenceState] =
    useState<ThreadingPreference>(initialThreading);

  const setThreadingPreference = useCallback(
    (mode: ThreadingPreference) => {
      if (mode === threadingPreference) return;
      try {
        sessionStorage.setItem("edgefit.threading", mode);
      } catch {
        // Storage nicht verfügbar → trotzdem reloaden, Init nutzt Default
      }
      // Hot-Swap des TFLite-Pthread-Pools ist nicht supported → harter Reload.
      window.location.reload();
      // Fallback-State-Update (Reload sollte synchron kommen)
      setThreadingPreferenceState(mode);
    },
    [threadingPreference],
  );
  const [activeThreadingMode, setActiveThreadingMode] = useState<
    "single" | "multi" | "unknown"
  >("unknown");
  const [activeNumThreads, setActiveNumThreads] = useState<number | null>(null);
  const [metrics, setMetrics] = useState<PoseMetrics>({
    inferenceTime: 0,
    fps: 0,
    frameCount: 0,
  });

  // Refs für FPS / Warmup / Backend-Init
  const frameTimestamps = useRef<number[]>([]);
  const frameCountRef = useRef(0);
  const warmupCounterRef = useRef(0);
  const backendReadyRef = useRef(false);
  const levelRef = useRef<QuantizationLevel>(initialLevel);
  const lastMeasurementRef = useRef<LastMeasurement | null>(null);
  // modelLoadMs-Messung: loadStart bei loadModel gesetzt, Wert beim ersten
  // NON-Warmup-Frame gestempelt (siehe detectPose / beginWarmup).
  const loadStartRef = useRef<number>(0);
  const modelLoadMsRef = useRef<number | null>(null);

  /**
   * Vollständiger Backend-Reset: verwirft Variablen, leert Engine-State.
   * Aufrufen zwischen Quantisierungsstufen, damit Cache-/JIT-Carryover
   * die Latenz-Messung nicht verzerrt.
   * Nutzt die nächste loadModel()-Inferenz für Re-Initialisierung.
   */
  const resetBackend = useCallback(async () => {
    try {
      // Alten Detector explizit freigeben — sonst hält die TFLite-Runtime
      // ggf. Referenzen, die den nächsten Modell-Load deadlocken (siehe
      // Threading-Audit: alpha.9 ist page-lifetime-singleton).
      setDetector((prev) => {
        if (prev) {
          try {
            prev.cleanUp?.();
            prev.dispose?.();
          } catch (e) {
            console.warn("Detector cleanup fehlgeschlagen:", e);
          }
        }
        return null;
      });
      tf.disposeVariables();
      tf.engine().reset();
      // Re-Init beim nächsten loadModel erzwingen
      backendReadyRef.current = false;
      // Kleine Pause, damit TFLite-Runtime interne Buffer freigeben kann
      // bevor das nächste Modell geladen wird (empirisch, keine offizielle API).
      await new Promise((resolve) => setTimeout(resolve, 50));
    } catch (e) {
      console.error("Backend-Reset fehlgeschlagen:", e);
    }
  }, []);

  /**
   * Startet das Warmup-Fenster neu: die ersten WARMUP_FRAMES Inferenzen der
   * kommenden Aufnahme werden als isWarmup=true markiert (Postprocessing
   * filtert sie heraus). MUSS bei Aufnahmestart aufgerufen werden — sonst
   * verbraucht der Setup/Countdown-Preview-Loop das Warmup-Budget bereits
   * vor der Aufnahme, und kein aufgezeichneter Frame trägt das Flag.
   * Setzt zugleich Frame-Zähler und FPS-Fenster zurück, damit frameIndex
   * und fps pro Aufnahme sauber bei null beginnen.
   *
   * Bewusst KEIN setIsWarmingUp(true): der UI-"Bereit/Warmup"-State gehört
   * loadModel (Modell-Aufwärmen). Würde beginWarmup ihn auf true setzen,
   * bliebe er nach der Aufnahme hängen (nur loadModel setzt ihn false) und
   * würde Export + Stufen-/Seiten-/Threading-Controls dauerhaft sperren.
   * Das Per-Frame-isWarmup-Flag (warmupCounterRef) markiert die Frames im
   * Export unabhängig vom UI-State.
   */
  const beginWarmup = useCallback(() => {
    warmupCounterRef.current = 0;
    frameCountRef.current = 0;
    frameTimestamps.current = [];
    // modelLoadMs neu messen: der erste NON-Warmup-Frame dieser Aufnahme
    // stempelt (jetzt − loadStart). loadStart wurde in loadModel gesetzt.
    modelLoadMsRef.current = null;
  }, []);

  // Modell laden (auch bei Level-Wechsel aufrufbar)
  const loadModel = useCallback(
    async (level: QuantizationLevel) => {
      try {
        // modelLoadMs-Start: Fetch/Init dieser Stufe (Vertrag: Fetch-Start).
        loadStartRef.current = performance.now();
        setIsLoading(true);
        setError(null);
        setIsWarmingUp(true);
        warmupCounterRef.current = 0;
        frameTimestamps.current = [];
        frameCountRef.current = 0;

        // Backend nur einmal initialisieren
        if (!backendReadyRef.current) {
          // Threading-Hint für tfjs-backend-wasm setzen (beeinflusst nur tfjs-eigene
          // Ops, NICHT die TFLite-Inferenz — diese läuft über tfjs-tflite und
          // wird unten via loadTFLiteModel({numThreads}) konfiguriert).
          const wantMulti =
            threadingPreference === "multi" && multiThreadingAvailable;
          tf.env().set("WASM_HAS_MULTITHREAD_SUPPORT", wantMulti);
          tf.env().set("WASM_HAS_SIMD_SUPPORT", true);

          // WASM-Binaries vom jsDelivr-CDN laden (Vite serviert sie sonst nicht)
          setWasmPaths(
            "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.22.0/dist/",
          );
          await tf.setBackend("wasm");
          await tf.ready();

          // window.tf MUSS gesetzt sein, BEVOR das tfjs-tflite UMD lädt
          // — die UMD-Factory captured tf bei der ersten Auswertung.
          (window as unknown as { tf: typeof tf }).tf = tf;

          // tfjs-tflite UMD dynamisch nachladen
          await loadTFLiteScript();

          backendReadyRef.current = true;
          console.log(
            "TensorFlow.js Backend:",
            tf.getBackend(),
            "crossOriginIsolated:",
            multiThreadingAvailable,
          );
        }

        // Altes Modell verwerfen
        setDetector((prev) => {
          if (prev) {
            try {
              // TFLiteModel stellt keine offizielle dispose-Methode bereit,
              // GC übernimmt — aber Referenz freigeben
            } catch {
              // ignore
            }
          }
          return null;
        });

        if (!window.tflite) {
          throw new Error(
            "window.tflite nicht verfügbar — UMD-Script in index.html prüfen",
          );
        }

        // tfjs-tflite hat eigene WASM-Binaries (getrennt vom backend-wasm).
        // Pfad einmalig auf jsDelivr setzen, damit _malloc verfügbar wird.
        if (window.tflite.setWasmPath) {
          // alpha.9 enthält die .wasm-Binaries; alpha.10 nicht
          window.tflite.setWasmPath(
            "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.9/dist/",
          );
        }
        // numThreads für TFLite-Runtime — nur wenn Multi-Threading wirklich
        // verfügbar UND vom Nutzer gewünscht. Sonst single-thread.
        // Cap bei 4: tfjs-MT-Speedup ist sublinear, höhere Werte bringen
        // wegen Synchronisations-Overhead bei kleinen Modellen nichts.
        const useMulti =
          threadingPreference === "multi" && multiThreadingAvailable;
        const numThreads = useMulti
          ? Math.min(navigator.hardwareConcurrency || 4, 4)
          : 1;

        // Modell-Fingerprint parallel berechnen (zweiter fetch trifft Browser-Cache).
        // Fail-Safe: falls TFLite mit numThreads scheitert (z.B. weil Pthread-Pool
        // nicht initialisiert werden kann), Retry mit numThreads=1.
        let model: TFLiteModelLike;
        let actuallyMulti = useMulti;
        try {
          model = await window.tflite.loadTFLiteModel(
            TFLITE_MODEL_URLS[level],
            {
              numThreads,
            },
          );
        } catch (e) {
          if (useMulti) {
            console.warn(
              "TFLite Multi-Thread-Load fehlgeschlagen, Fallback auf Single:",
              e,
            );
            actuallyMulti = false;
            model = await window.tflite.loadTFLiteModel(
              TFLITE_MODEL_URLS[level],
              {
                numThreads: 1,
              },
            );
          } else {
            throw e;
          }
        }
        const fingerprint = await fingerprintModel(
          TFLITE_MODEL_URLS[level],
        ).catch((e) => {
          console.warn("Fingerprint fehlgeschlagen:", e);
          return null;
        });

        // activeThreadingMode jetzt anhand des TATSÄCHLICH genutzten Pfads setzen,
        // nicht anhand der vorher gesetzten Flag (tautologisch).
        const realNumThreads = actuallyMulti ? numThreads : 1;
        setActiveThreadingMode(actuallyMulti ? "multi" : "single");
        setActiveNumThreads(realNumThreads);

        levelRef.current = level;
        setCurrentLevel(level);
        setModelFingerprint(fingerprint);
        setDetector(model);
        setIsLoading(false);

        // Modell ist geladen — UI-State "einsatzbereit" sofort setzen.
        // Der per-Frame isWarmup-Flag (warmupCounterRef in detectPose) markiert
        // die ersten WARMUP_FRAMES realen Inferenzen weiterhin als Warmup im
        // JSON-Export. Postprocessing in Python filtert diese aus.
        setIsWarmingUp(false);
        console.log(`TFLite-Modell geladen (${level})`);
      } catch (err) {
        const message =
          err instanceof Error
            ? err.message
            : "Modell konnte nicht geladen werden";
        setError(message);
        setIsLoading(false);
        console.error("Modell-Initialisierung fehlgeschlagen:", err);
      }
      // Audit-Fix: leere Deps lasen stale threadingPreference + multiThreadingAvailable.
      // Mit Reload-on-Toggle ist beides effektiv konstant pro Session — die Deps
      // sind trotzdem korrekt deklariert, damit React-Hook-Lint-Regel erfüllt ist.
    },
    [threadingPreference, multiThreadingAvailable],
  );

  // Initiales Laden
  useEffect(() => {
    let mounted = true;
    (async () => {
      if (mounted) {
        await loadModel(initialLevel);
      }
    })();
    return () => {
      mounted = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /**
   * Führt Pose-Detection auf einem Video-Frame durch.
   * Disposed inputTensor & outputTensor sauber pro Frame (Memory-Leak-Schutz).
   */
  const detectPose = useCallback(
    async (
      video: HTMLVideoElement,
      opts?: { record?: boolean },
    ): Promise<DetectResult> => {
      if (!detector || video.readyState < 2) {
        return {
          pose: null,
          reason: "other",
          message: "Detector/Video nicht bereit",
        };
      }

      const record = opts?.record ?? true;
      const level = levelRef.current;
      let outputTensor: tf.Tensor | null = null;

      try {
        // Eingabe-Tensor bauen — tf.tidy verwirft Zwischen-Tensoren automatisch
        const inputTensor = tf.tidy(() => {
          const pixels = tf.browser.fromPixels(video);
          const resized = tf.image.resizeBilinear(pixels, [
            MODEL_INPUT_SIZE,
            MODEL_INPUT_SIZE,
          ]);
          // MoveNet Kaggle-Modelle: Input-dtype ist uint8 für BEIDE fp16 UND int8,
          // nur fp32 erwartet float32. Die Bezeichnungen fp16/int8 beziehen sich
          // auf die Quantisierung der GEWICHTE, nicht auf den Input-Typ.
          // resizeBilinear liefert float32 — bei uint8-Input zurück casten.
          const needsIntInput = level === "int8" || level === "fp16";
          const typed = needsIntInput ? tf.cast(resized, "int32") : resized;
          return tf.expandDims(typed, 0);
        });

        const t0 = performance.now();
        outputTensor = detector.predict(inputTensor) as tf.Tensor;
        // Warten bis GPU/WASM-Arbeit fertig — GPU/WASM-Sync ist der eigentliche
        // Readback-Punkt; eigener catch für Vertrag-Grund "readback_error".
        try {
          await outputTensor.data();
        } catch (readbackErr) {
          inputTensor.dispose();
          if (outputTensor) {
            try {
              outputTensor.dispose();
            } catch {
              // ignore
            }
            outputTensor = null;
          }
          const message =
            readbackErr instanceof Error
              ? readbackErr.message
              : String(readbackErr);
          console.error("Readback-Fehler:", readbackErr);
          setError(`Readback-Fehler (${level}): ${message}`);
          return { pose: null, reason: "readback_error", message };
        }
        const inferenceMs = performance.now() - t0;

        // Input nach Inferenz freigeben
        inputTensor.dispose();

        const raw = outputTensor.dataSync();
        const vw = video.videoWidth || MODEL_INPUT_SIZE;
        const vh = video.videoHeight || MODEL_INPUT_SIZE;

        // Output [1,1,17,3] flattened: i*3 = y, +1 = x, +2 = score
        const keypoints: poseDetection.Keypoint[] = [];
        for (let i = 0; i < 17; i++) {
          const y = raw[i * 3];
          const x = raw[i * 3 + 1];
          const score = raw[i * 3 + 2];
          keypoints.push({
            x: x * vw,
            y: y * vh,
            score,
            name: KEYPOINT_NAMES[i],
          });
        }

        outputTensor.dispose();
        outputTensor = null;

        const pose: Pose = { keypoints };
        setPoses([pose]);

        // Setup-Preview (record=false) darf die Messzähler NICHT verbrauchen —
        // sonst stiehlt eine Preview-Inferenz Warmup-/Frame-Budget der Aufnahme.
        if (!record) {
          return { pose };
        }

        // Per-Frame Warmup-Flag — markiert die ersten WARMUP_FRAMES Inferenzen
        // der Aufnahme für das Postprocessing-Filter in Python.
        // UI-State (isWarmingUp) wird NICHT mehr hier gesetzt — siehe loadModel.
        warmupCounterRef.current += 1;
        const inWarmup = warmupCounterRef.current <= WARMUP_FRAMES;

        // modelLoadMs einmal pro Aufnahme stempeln: erster valider (Non-Warmup)
        // Frame markiert "Modell einsatzbereit" (Vertrag: bis erstes predict
        // nach Warmup).
        if (!inWarmup && modelLoadMsRef.current === null) {
          modelLoadMsRef.current = performance.now() - loadStartRef.current;
        }

        // FPS — gleitender Durchschnitt der letzten 30 Frames
        const now = performance.now();
        frameTimestamps.current.push(now);
        if (frameTimestamps.current.length > 30) {
          frameTimestamps.current.shift();
        }
        let fps = 0;
        if (frameTimestamps.current.length > 1) {
          const timeSpan = now - frameTimestamps.current[0];
          fps = (frameTimestamps.current.length - 1) / (timeSpan / 1000);
        }

        frameCountRef.current += 1;

        // Synchroner Snapshot für Frame-Recording (App.tsx liest daraus)
        lastMeasurementRef.current = {
          inferenceMs,
          fps,
          frameIndex: frameCountRef.current,
          timestampMs: t0,
          isWarmup: inWarmup,
        };

        // Inferenzzeit erst nach Warmup in State schreiben
        if (!inWarmup) {
          setMetrics({
            inferenceTime: Math.round(inferenceMs * 100) / 100,
            fps: Math.round(fps * 10) / 10,
            frameCount: frameCountRef.current,
          });
        } else {
          setMetrics((m) => ({ ...m, frameCount: frameCountRef.current }));
        }

        return { pose };
      } catch (err) {
        // Sicherheitsnetz — Output disposen falls Fehler nach predict.
        // predict-Bau/Tensor-Fehler -> Vertrag-Grund "predict_error"
        // (Readback wird oben separat als "readback_error" behandelt).
        if (outputTensor) {
          try {
            outputTensor.dispose();
          } catch {
            // ignore
          }
        }
        const message = err instanceof Error ? err.message : String(err);
        console.error("Pose Detection Fehler:", err);
        // Sichtbar ins UI: sonst bleiben Frames leer im Export ohne Hinweis
        setError(`Inferenz-Fehler (${level}): ${message}`);
        return { pose: null, reason: "predict_error", message };
      }
    },
    [detector],
  );

  return {
    detector,
    poses,
    metrics,
    isLoading,
    error,
    detectPose,
    loadModel,
    resetBackend,
    beginWarmup,
    currentLevel,
    isWarmingUp,
    lastMeasurementRef,
    modelLoadMsRef,
    modelFingerprint,
    threadingPreference,
    setThreadingPreference,
    multiThreadingAvailable,
    activeThreadingMode,
    activeNumThreads,
  };
}
