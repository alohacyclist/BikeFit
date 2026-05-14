import { useState, useEffect, useRef, useCallback } from "react";
import * as poseDetection from "@tensorflow-models/pose-detection";
import * as tf from "@tensorflow/tfjs-core";
import { setWasmPaths } from "@tensorflow/tfjs-backend-wasm";

// tfjs-tflite wird als UMD-Script in index.html geladen (ESM-Import scheitert
// wegen fehlender Submodule). Stellt window.tflite bereit.
interface TFLiteModelLike {
  predict: (input: tf.Tensor | tf.Tensor[]) => tf.Tensor | tf.Tensor[];
}
interface TFLiteGlobal {
  loadTFLiteModel: (url: string) => Promise<TFLiteModelLike>;
  setWasmPath?: (path: string) => void;
}
declare global {
  interface Window {
    tflite?: TFLiteGlobal;
  }
}
import {
  QuantizationLevel,
  TFLITE_MODEL_URLS,
  WARMUP_FRAMES,
  MODEL_INPUT_SIZE,
  KEYPOINT_NAMES,
} from "../types/quantization";

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
  isWarmup: boolean;
}

export interface UsePoseDetectionResult {
  detector: TFLiteModelLike | null;
  poses: Pose[];
  metrics: PoseMetrics;
  isLoading: boolean;
  error: string | null;
  detectPose: (video: HTMLVideoElement) => Promise<Pose | null>;
  loadModel: (level: QuantizationLevel) => Promise<void>;
  currentLevel: QuantizationLevel;
  isWarmingUp: boolean;
  lastMeasurementRef: React.MutableRefObject<LastMeasurement | null>;
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

  // Modell laden (auch bei Level-Wechsel aufrufbar)
  const loadModel = useCallback(async (level: QuantizationLevel) => {
    try {
      setIsLoading(true);
      setError(null);
      setIsWarmingUp(true);
      warmupCounterRef.current = 0;
      frameTimestamps.current = [];
      frameCountRef.current = 0;

      // Backend nur einmal initialisieren
      if (!backendReadyRef.current) {
        // WASM-Binaries vom jsDelivr-CDN laden (Vite serviert sie sonst nicht)
        setWasmPaths(
          "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.22.0/dist/",
        );
        await tf.setBackend("wasm");
        await tf.ready();
        backendReadyRef.current = true;
        console.log("TensorFlow.js Backend:", tf.getBackend());
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
        window.tflite.setWasmPath(
          "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.10/dist/",
        );
      }
      const model = await window.tflite.loadTFLiteModel(
        TFLITE_MODEL_URLS[level],
      );

      levelRef.current = level;
      setCurrentLevel(level);
      setDetector(model);
      setIsLoading(false);
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
  }, []);

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
    async (video: HTMLVideoElement): Promise<Pose | null> => {
      if (!detector || video.readyState < 2) {
        return null;
      }

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
          // INT8 erwartet uint8; FP32/FP16 erwarten float32.
          // resizeBilinear liefert float32 — bei int8 wieder zurück casten.
          const typed = level === "int8" ? tf.cast(resized, "int32") : resized;
          return tf.expandDims(typed, 0);
        });

        const t0 = performance.now();
        outputTensor = detector.predict(inputTensor) as tf.Tensor;
        // Warten bis GPU/WASM-Arbeit fertig
        await outputTensor.data();
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

        // Warmup-Phase
        warmupCounterRef.current += 1;
        const inWarmup = warmupCounterRef.current <= WARMUP_FRAMES;
        if (!inWarmup && isWarmingUp) {
          setIsWarmingUp(false);
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
          isWarmup: inWarmup,
        };

        setPoses([pose]);
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

        return pose;
      } catch (err) {
        // Sicherheitsnetz — Output disposen falls Fehler nach predict
        if (outputTensor) {
          try {
            outputTensor.dispose();
          } catch {
            // ignore
          }
        }
        console.error("Pose Detection Fehler:", err);
        return null;
      }
    },
    [detector, isWarmingUp],
  );

  return {
    detector,
    poses,
    metrics,
    isLoading,
    error,
    detectPose,
    loadModel,
    currentLevel,
    isWarmingUp,
    lastMeasurementRef,
  };
}
