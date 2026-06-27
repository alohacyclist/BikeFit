/**
 * BenchmarkExporter — sammelt Messdaten pro Frame und exportiert sie als JSON
 * für Postprocessing in Python (NumPy / pandas / SciPy).
 */

import { QuantizationLevel } from "../types/quantization";
import type { BodySide } from "../utils/AngleCalculator";
import type { ModelFingerprint } from "../utils/modelFingerprint";

export interface FrameMeasurement {
  frameIndex: number;
  timestampMs: number;
  inferenceMs: number;
  fps: number;
  /**
   * Roh-Kniewinkel — kein App-seitiger Confidence-Filter. Selektion erfolgt
   * im Python-Postprocessing anhand der `keypointScores` mit dort
   * dokumentierter Schwelle (Sensitivity-Sweep).
   */
  kneeAngleRight: number;
  kneeAngleLeft: number;
  keypointScores: number[];
  isWarmup: boolean;
}

interface SystemInfo {
  userAgent: string;
  hardwareConcurrency: number;
  deviceMemory?: number;
  /** Vom Browser gemeldete Cross-Origin-Isolation (Voraussetzung für SAB). */
  crossOriginIsolated: boolean;
  /** Ob SharedArrayBuffer im aktuellen Kontext definiert ist. */
  sharedArrayBufferAvailable: boolean;
}

export interface BenchmarkSession {
  participantId: string;
  quantizationLevel: QuantizationLevel;
  startTimestamp: string;
  systemInfo: SystemInfo;
  warmupFrames: number;
  lockedSide: BodySide | null; // welche Körperseite getrackt wurde
  modelFingerprint: ModelFingerprint | null;
  threadingMode: "single" | "multi" | "unknown";
  /** Tatsächlich an TFLite übergebener numThreads-Parameter. */
  numThreads: number | null;
  videoSource: "file";
  videoSourceName?: string;
  /** Ziel-Framerate des deterministischen Replay-Steppings (CFR-Annahme). */
  targetFps: number;
  /** Länge des Quell-Videos in Sekunden (0 bis Aufnahme finalisiert). */
  durationSeconds: number;
  /** Erwartete Frame-Anzahl = floor(durationSeconds * targetFps). */
  expectedFrames: number;
  frames: FrameMeasurement[];
}

import { WARMUP_FRAMES } from "../types/quantization";

function collectSystemInfo(): SystemInfo {
  const navAny = navigator as Navigator & { deviceMemory?: number };
  const coi =
    typeof self !== "undefined" &&
    (self as unknown as { crossOriginIsolated?: boolean })
      .crossOriginIsolated === true;
  return {
    userAgent: navigator.userAgent,
    hardwareConcurrency: navigator.hardwareConcurrency ?? 0,
    deviceMemory: navAny.deviceMemory,
    crossOriginIsolated: coi,
    sharedArrayBufferAvailable: typeof SharedArrayBuffer !== "undefined",
  };
}

export class BenchmarkExporter {
  private session: BenchmarkSession | null = null;

  startSession(participantId: string, level: QuantizationLevel): void {
    this.session = {
      participantId,
      quantizationLevel: level,
      startTimestamp: new Date().toISOString(),
      systemInfo: collectSystemInfo(),
      warmupFrames: WARMUP_FRAMES,
      lockedSide: null,
      modelFingerprint: null,
      threadingMode: "unknown",
      numThreads: null,
      videoSource: "file",
      targetFps: 0,
      durationSeconds: 0,
      expectedFrames: 0,
      frames: [],
    };
  }

  setNumThreads(n: number | null): void {
    if (!this.session) return;
    this.session.numThreads = n;
  }

  setLockedSide(side: BodySide): void {
    if (!this.session) return;
    this.session.lockedSide = side;
  }

  setModelFingerprint(fp: ModelFingerprint | null): void {
    if (!this.session) return;
    this.session.modelFingerprint = fp;
  }

  setThreadingMode(mode: "single" | "multi" | "unknown"): void {
    if (!this.session) return;
    this.session.threadingMode = mode;
  }

  setVideoSource(name: string | undefined, targetFps: number): void {
    if (!this.session) return;
    this.session.videoSourceName = name;
    this.session.targetFps = targetFps;
  }

  /** Beim Finalisieren gesetzt — Videolänge + erwartete Frame-Anzahl. */
  setVideoMeta(durationSeconds: number, expectedFrames: number): void {
    if (!this.session) return;
    this.session.durationSeconds = durationSeconds;
    this.session.expectedFrames = expectedFrames;
  }

  recordFrame(measurement: FrameMeasurement): void {
    if (!this.session) return;
    this.session.frames.push(measurement);
  }

  exportJSON(): string {
    if (!this.session) {
      return JSON.stringify({ error: "Keine aktive Session" });
    }
    return JSON.stringify(this.session, null, 2);
  }

  downloadJSON(): void {
    if (!this.session) return;

    const json = this.exportJSON();
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);

    const timestamp = this.session.startTimestamp.replace(/[:.]/g, "-");
    const filename = `benchmark_${this.session.participantId}_${this.session.quantizationLevel}_${timestamp}.json`;

    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  reset(): void {
    this.session = null;
  }

  hasSession(): boolean {
    return this.session !== null;
  }
}

export const benchmarkExporter = new BenchmarkExporter();
