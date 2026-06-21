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
  kneeAngleRight: number | null;
  kneeAngleLeft: number | null;
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

export interface ValidationMetrics {
  validKneeRatioRight: number; // 0..1 — Anteil non-warmup Frames mit gültigem rechtem Kniewinkel
  validKneeRatioLeft: number;
  validKneeRatioSelected: number; // bezogen auf lockedSide
  meanKeypointScores: number[]; // pro KP-Index, gemittelt über non-warmup Frames
  interpolatedFrames: number; // wie viele Frames per Hold-Last-Value überbrückt
  totalCyclesDetected: number;
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
  frames: FrameMeasurement[];
  validationMetrics?: ValidationMetrics;
}

export interface BenchmarkSummary {
  totalFrames: number;
  validFrames: number;
  meanInferenceMs: number;
  stdInferenceMs: number;
  p50Ms: number;
  p95Ms: number;
  meanFps: number;
  meanKneeAngleRight: number | null;
  meanKneeAngleLeft: number | null;
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

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  const idx = Math.min(
    sorted.length - 1,
    Math.max(0, Math.floor((p / 100) * sorted.length)),
  );
  return sorted[idx];
}

function mean(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function std(values: number[], avg: number): number {
  if (values.length === 0) return 0;
  const variance =
    values.reduce((acc, v) => acc + (v - avg) * (v - avg), 0) / values.length;
  return Math.sqrt(variance);
}

function meanOrNull(values: Array<number | null>): number | null {
  const filtered = values.filter((v): v is number => v !== null);
  if (filtered.length === 0) return null;
  return mean(filtered);
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

  recordFrame(measurement: FrameMeasurement): void {
    if (!this.session) return;
    this.session.frames.push(measurement);
  }

  /**
   * Berechnet Validierungs-Aggregate aus den non-warmup Frames.
   * Wird beim Export aufgerufen und auf die Session geschrieben.
   */
  private computeValidationMetrics(
    interpolatedFrames: number,
    totalCyclesDetected: number,
  ): ValidationMetrics {
    if (!this.session) {
      return {
        validKneeRatioRight: 0,
        validKneeRatioLeft: 0,
        validKneeRatioSelected: 0,
        meanKeypointScores: [],
        interpolatedFrames: 0,
        totalCyclesDetected: 0,
      };
    }
    const valid = this.session.frames.filter((f) => !f.isWarmup);
    const n = valid.length || 1;
    const validRight =
      valid.filter((f) => f.kneeAngleRight !== null).length / n;
    const validLeft = valid.filter((f) => f.kneeAngleLeft !== null).length / n;
    const side = this.session.lockedSide;
    const validSel =
      side === "right" ? validRight : side === "left" ? validLeft : 0;

    // Mean keypoint scores: bilde Spaltenmittel über 17 KPs
    const numKps = valid[0]?.keypointScores.length ?? 17;
    const meanScores = new Array(numKps).fill(0);
    for (const f of valid) {
      for (let i = 0; i < numKps; i++) {
        meanScores[i] += f.keypointScores[i] ?? 0;
      }
    }
    for (let i = 0; i < numKps; i++) meanScores[i] /= n;

    return {
      validKneeRatioRight: validRight,
      validKneeRatioLeft: validLeft,
      validKneeRatioSelected: validSel,
      meanKeypointScores: meanScores,
      interpolatedFrames,
      totalCyclesDetected,
    };
  }

  /**
   * Schreibt die Validierungsmetriken in die aktive Session.
   * Aufrufen kurz vor Export, mit aktuellen Werten aus Analyzer.
   */
  finalizeMetrics(interpolatedFrames: number, cyclesDetected: number): void {
    if (!this.session) return;
    this.session.validationMetrics = this.computeValidationMetrics(
      interpolatedFrames,
      cyclesDetected,
    );
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

  getSummary(): BenchmarkSummary {
    const empty: BenchmarkSummary = {
      totalFrames: 0,
      validFrames: 0,
      meanInferenceMs: 0,
      stdInferenceMs: 0,
      p50Ms: 0,
      p95Ms: 0,
      meanFps: 0,
      meanKneeAngleRight: null,
      meanKneeAngleLeft: null,
    };

    if (!this.session) return empty;

    const all = this.session.frames;
    const valid = all.filter((f) => !f.isWarmup);

    if (valid.length === 0) {
      return { ...empty, totalFrames: all.length };
    }

    const inferenceTimes = valid.map((f) => f.inferenceMs);
    const sorted = [...inferenceTimes].sort((a, b) => a - b);
    const meanInf = mean(inferenceTimes);

    return {
      totalFrames: all.length,
      validFrames: valid.length,
      meanInferenceMs: meanInf,
      stdInferenceMs: std(inferenceTimes, meanInf),
      p50Ms: percentile(sorted, 50),
      p95Ms: percentile(sorted, 95),
      meanFps: mean(valid.map((f) => f.fps)),
      meanKneeAngleRight: meanOrNull(valid.map((f) => f.kneeAngleRight)),
      meanKneeAngleLeft: meanOrNull(valid.map((f) => f.kneeAngleLeft)),
    };
  }

  reset(): void {
    this.session = null;
  }

  hasSession(): boolean {
    return this.session !== null;
  }
}

export const benchmarkExporter = new BenchmarkExporter();
