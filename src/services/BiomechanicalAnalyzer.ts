/**
 * BiomechanicalAnalyzer — Pedalzyklus-Erkennung über den Kniewinkel.
 *
 * Scope: Kniewinkel-only. Die FSM erkennt Pedalzyklen aus geglätteten
 * Kniewinkel-Verläufen; Statistiken werden ausschließlich für das Knie
 * berechnet.
 */

import {
  BiomechanicalAngles,
  KNEE_EXTENSION_RANGE,
  KNEE_FLEXION_RANGE,
  evaluateAngle,
  AngleStatus,
} from '../utils/AngleCalculator';

export interface FrameData {
  timestamp: number;
  angles: BiomechanicalAngles;
  confidence: number;
}

export interface PedalCycle {
  startTime: number;
  endTime: number;
  kneeMin: number;
  kneeMax: number;
  frames: FrameData[];
}

export interface AngleStatistics {
  min: number;
  max: number;
  average: number;
  stdDev: number;
  samples: number;
}

export interface AnalysisResults {
  cycleCount: number;
  duration: number;
  timestamp: Date;
  statistics: {
    kneeExtension: AngleStatistics;
    kneeFlexion: AngleStatistics;
  };
  evaluations: {
    kneeExtension: AngleStatus;
    kneeFlexion: AngleStatus;
  };
  rawCycles: PedalCycle[];
}

export interface AnalyzerConfig {
  targetCycles: number;
  minCycleFrames: number;
  minConfidence: number;
  kneeThresholdHigh: number;
  kneeThresholdLow: number;
  smoothingWindow: number;
}

const DEFAULT_CONFIG: AnalyzerConfig = {
  targetCycles: 5,
  minCycleFrames: 10,
  minConfidence: 0.3,
  kneeThresholdHigh: 135,
  kneeThresholdLow: 115,
  smoothingWindow: 5,
};

const MAX_INTERPOLATED_FRAMES = 5;

export type CyclePhase = 'searching' | 'flexion' | 'extension';

export class BiomechanicalAnalyzer {
  private config: AnalyzerConfig;
  private frames: FrameData[] = [];
  private cycles: PedalCycle[] = [];
  private currentPhase: CyclePhase = 'searching';
  private cycleStartIndex: number = 0;
  private smoothedKneeAngles: number[] = [];
  private lastValidKneeAngle: number | null = null;
  private interpolatedFrameCount: number = 0;
  private interpolatedFramesTotal: number = 0;

  constructor(config: Partial<AnalyzerConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  addFrame(angles: BiomechanicalAngles, confidence: number): void {
    const frame: FrameData = {
      timestamp: performance.now(),
      angles,
      confidence,
    };
    this.frames.push(frame);

    let kneeForDetect: number | null = null;
    if (angles.knee !== null && confidence >= this.config.minConfidence) {
      kneeForDetect = angles.knee;
      this.lastValidKneeAngle = angles.knee;
      this.interpolatedFrameCount = 0;
    } else if (
      this.lastValidKneeAngle !== null &&
      this.interpolatedFrameCount < MAX_INTERPOLATED_FRAMES
    ) {
      kneeForDetect = this.lastValidKneeAngle;
      this.interpolatedFrameCount += 1;
      this.interpolatedFramesTotal += 1;
    }
    if (kneeForDetect !== null) this.detectCycle(kneeForDetect);
  }

  getInterpolatedFramesCount(): number {
    return this.interpolatedFramesTotal;
  }

  private detectCycle(kneeAngle: number): void {
    this.smoothedKneeAngles.push(kneeAngle);
    if (this.smoothedKneeAngles.length > this.config.smoothingWindow) {
      this.smoothedKneeAngles.shift();
    }
    const smoothed =
      this.smoothedKneeAngles.reduce((a, b) => a + b, 0) /
      this.smoothedKneeAngles.length;

    switch (this.currentPhase) {
      case 'searching':
        if (smoothed > this.config.kneeThresholdHigh) {
          this.currentPhase = 'extension';
          this.cycleStartIndex = this.frames.length - 1;
        }
        break;
      case 'extension':
        if (smoothed < this.config.kneeThresholdLow) {
          this.currentPhase = 'flexion';
        }
        break;
      case 'flexion':
        if (smoothed > this.config.kneeThresholdHigh) {
          this.completeCycle();
          this.cycleStartIndex = this.frames.length - 1;
          this.currentPhase = 'extension';
        }
        break;
    }
  }

  private completeCycle(): void {
    const cycleFrames = this.frames.slice(this.cycleStartIndex);
    if (cycleFrames.length < this.config.minCycleFrames) return;
    const kneeAngles = cycleFrames
      .map((f) => f.angles.knee)
      .filter((a): a is number => a !== null);
    if (kneeAngles.length === 0) return;
    this.cycles.push({
      startTime: cycleFrames[0].timestamp,
      endTime: cycleFrames[cycleFrames.length - 1].timestamp,
      kneeMin: Math.min(...kneeAngles),
      kneeMax: Math.max(...kneeAngles),
      frames: cycleFrames,
    });
  }

  getCycleCount(): number {
    return this.cycles.length;
  }
  getProgress(): number {
    return Math.min(1, this.cycles.length / this.config.targetCycles);
  }
  isComplete(): boolean {
    return this.cycles.length >= this.config.targetCycles;
  }
  getCurrentPhase(): CyclePhase {
    return this.currentPhase;
  }

  private calculateStatistics(values: number[]): AngleStatistics {
    if (values.length === 0)
      return { min: 0, max: 0, average: 0, stdDev: 0, samples: 0 };
    const min = Math.min(...values);
    const max = Math.max(...values);
    const average = values.reduce((a, b) => a + b, 0) / values.length;
    const squaredDiffs = values.map((v) => Math.pow(v - average, 2));
    const variance =
      squaredDiffs.reduce((a, b) => a + b, 0) / values.length;
    return {
      min,
      max,
      average,
      stdDev: Math.sqrt(variance),
      samples: values.length,
    };
  }

  getResults(): AnalysisResults | null {
    if (this.cycles.length === 0) return null;
    const kneeMaxValues = this.cycles.map((c) => c.kneeMax);
    const kneeMinValues = this.cycles.map((c) => c.kneeMin);
    const kneeExt = this.calculateStatistics(kneeMaxValues);
    const kneeFlex = this.calculateStatistics(kneeMinValues);
    const first = this.cycles[0];
    const last = this.cycles[this.cycles.length - 1];
    return {
      cycleCount: this.cycles.length,
      duration: last.endTime - first.startTime,
      timestamp: new Date(),
      statistics: { kneeExtension: kneeExt, kneeFlexion: kneeFlex },
      evaluations: {
        kneeExtension: evaluateAngle(kneeExt.average, KNEE_EXTENSION_RANGE),
        kneeFlexion: evaluateAngle(kneeFlex.average, KNEE_FLEXION_RANGE),
      },
      rawCycles: this.cycles,
    };
  }

  reset(): void {
    this.frames = [];
    this.cycles = [];
    this.currentPhase = 'searching';
    this.cycleStartIndex = 0;
    this.smoothedKneeAngles = [];
    this.lastValidKneeAngle = null;
    this.interpolatedFrameCount = 0;
    this.interpolatedFramesTotal = 0;
  }

  updateConfig(config: Partial<AnalyzerConfig>): void {
    this.config = { ...this.config, ...config };
  }
}

let analyzerInstance: BiomechanicalAnalyzer | null = null;

export function getAnalyzer(
  config?: Partial<AnalyzerConfig>
): BiomechanicalAnalyzer {
  if (!analyzerInstance) {
    analyzerInstance = new BiomechanicalAnalyzer(config);
  }
  return analyzerInstance;
}

export function resetAnalyzer(): void {
  if (analyzerInstance) analyzerInstance.reset();
  analyzerInstance = null;
}
