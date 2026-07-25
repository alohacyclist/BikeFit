/**
 * BenchmarkExporter — sammelt Messdaten pro Frame und exportiert sie als JSON
 * nach dem Vertrag v1.0.0 (siehe scripts/postprocess_benchmark.py).
 *
 * Reiner Datensammler: keine App-seitige Confidence-Filterung, keine
 * Statistik. Winkel werden pro Frame roh mitgeschrieben; Selektion,
 * Ausreißerfilter und Aggregation passieren ausschließlich im
 * Python-Postprocessing.
 */

import { QuantizationLevel, WARMUP_FRAMES } from "../types/quantization";
import type { BodySide } from "../utils/AngleCalculator";
import { detectHardware, toCompactTimestamp } from "../utils/hardwareInfo";
import type { HardwareInfo } from "../utils/hardwareInfo";
import type { EnvironmentInfo } from "../utils/environment";

export const SCHEMA_VERSION = "1.1.0" as const;

export interface Point2D {
  x: number;
  y: number;
}

export interface FrameMeasurement {
  /** 0-basiert, entspricht i in seekTo((i+0.5)/targetFps). */
  frameIndex: number;
  /** performance.now() VOR predict. */
  timestampMs: number;
  /** predict + Readback (data()). */
  inferenceMs: number;
  /** 1000/inferenceMs, nur informativ. */
  fps: number;
  kneeAngleRight: number;
  kneeAngleLeft: number;
  /** Länge 17, COCO-Reihenfolge. */
  keypoints: Point2D[];
  /** Länge 17, [0..1]. */
  keypointScores: number[];
  isWarmup: boolean;
}

export type DroppedFrameReason =
  "seek_timeout" | "predict_error" | "readback_error" | "other";

export interface DroppedFrame {
  frameIndex: number;
  reason: DroppedFrameReason;
  message?: string;
}

export interface BenchmarkSession {
  schemaVersion: typeof SCHEMA_VERSION;
  probandId: string;
  level: QuantizationLevel;
  runIndex: number;
  createdAt: string;
  targetFps: number;
  bodySide: BodySide;
  threading: "single" | "multi";
  videoDurationSec: number;
  videoTotalFrames: number;
  warmupCount: number;
  modelFingerprintSha256: string;
  modelUrl: string;
  modelLoadMs: number;
  userAgent: string;
  hardware: HardwareInfo;
  /** v1.1.0: Umgebungs-/Reproduzierbarkeitsblock (optional bei Altdaten 1.0.0). */
  environment?: EnvironmentInfo;
  frames: FrameMeasurement[];
  droppedFrames: DroppedFrame[];
}

export class BenchmarkExporter {
  private session: BenchmarkSession | null = null;

  startSession(
    probandId: string,
    level: QuantizationLevel,
    runIndex: number,
  ): void {
    this.session = {
      schemaVersion: SCHEMA_VERSION,
      probandId,
      level,
      runIndex,
      createdAt: new Date().toISOString(),
      targetFps: 30,
      bodySide: "left",
      threading: "single",
      videoDurationSec: 0,
      videoTotalFrames: 0,
      warmupCount: WARMUP_FRAMES,
      modelFingerprintSha256: "",
      modelUrl: "",
      modelLoadMs: 0,
      userAgent:
        typeof navigator !== "undefined" ? navigator.userAgent : "unknown",
      hardware: detectHardware(),
      frames: [],
      droppedFrames: [],
    };
  }

  setBodySide(side: BodySide): void {
    if (this.session) this.session.bodySide = side;
  }

  setThreading(mode: "single" | "multi"): void {
    if (this.session) this.session.threading = mode;
  }

  setModel(sha256: string, url: string): void {
    if (!this.session) return;
    this.session.modelFingerprintSha256 = sha256;
    this.session.modelUrl = url;
  }

  setModelLoadMs(ms: number): void {
    if (this.session) this.session.modelLoadMs = ms;
  }

  setHardware(hardware: HardwareInfo): void {
    if (this.session) this.session.hardware = hardware;
  }

  setEnvironment(environment: EnvironmentInfo): void {
    if (this.session) this.session.environment = environment;
  }

  setTargetFps(fps: number): void {
    if (this.session) this.session.targetFps = fps;
  }

  /** Beim Finalisieren gesetzt — Videolänge + Gesamt-Frame-Zahl (== frames + dropped). */
  setVideoMeta(durationSec: number, totalFrames: number): void {
    if (!this.session) return;
    this.session.videoDurationSec = durationSec;
    this.session.videoTotalFrames = totalFrames;
  }

  recordFrame(measurement: FrameMeasurement): void {
    if (this.session) this.session.frames.push(measurement);
  }

  recordDropped(dropped: DroppedFrame): void {
    if (this.session) this.session.droppedFrames.push(dropped);
  }

  exportJSON(): string {
    if (!this.session) {
      return JSON.stringify({ error: "Keine aktive Session" });
    }
    return JSON.stringify(this.session, null, 2);
  }

  /** Vertrag-konformer Dateiname der aktuellen Session (leer ohne Session). */
  getFilename(): string {
    if (!this.session) return "";
    const stamp = toCompactTimestamp(this.session.createdAt);
    return `benchmark_${this.session.probandId}_${this.session.level}_${stamp}.json`;
  }

  downloadJSON(): void {
    if (!this.session) return;

    const json = this.exportJSON();
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);

    const filename = this.getFilename();

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
