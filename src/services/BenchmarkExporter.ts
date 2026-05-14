/**
 * BenchmarkExporter — sammelt Messdaten pro Frame und exportiert sie als JSON
 * für Postprocessing in Python (NumPy / pandas / SciPy).
 */

import { QuantizationLevel } from '../types/quantization';

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
}

export interface BenchmarkSession {
  participantId: string;
  quantizationLevel: QuantizationLevel;
  startTimestamp: string;
  systemInfo: SystemInfo;
  warmupFrames: number;
  frames: FrameMeasurement[];
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

import { WARMUP_FRAMES } from '../types/quantization';

function collectSystemInfo(): SystemInfo {
  const navAny = navigator as Navigator & { deviceMemory?: number };
  return {
    userAgent: navigator.userAgent,
    hardwareConcurrency: navigator.hardwareConcurrency ?? 0,
    deviceMemory: navAny.deviceMemory,
  };
}

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  const idx = Math.min(
    sorted.length - 1,
    Math.max(0, Math.floor((p / 100) * sorted.length))
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
      frames: [],
    };
  }

  recordFrame(measurement: FrameMeasurement): void {
    if (!this.session) return;
    this.session.frames.push(measurement);
  }

  exportJSON(): string {
    if (!this.session) {
      return JSON.stringify({ error: 'Keine aktive Session' });
    }
    return JSON.stringify(this.session, null, 2);
  }

  downloadJSON(): void {
    if (!this.session) return;

    const json = this.exportJSON();
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    const timestamp = this.session.startTimestamp.replace(/[:.]/g, '-');
    const filename = `benchmark_${this.session.participantId}_${this.session.quantizationLevel}_${timestamp}.json`;

    const a = document.createElement('a');
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
