/**
 * BiomechanicalAnalyzer - Zyklus-Erkennung und Datenanalyse für Bike-Fitting
 *
 * Erkennt Pedalzyklen über Kniewinkel-Variationen und sammelt
 * biomechanische Daten für die Analyse.
 */

import {
  BiomechanicalAngles,
  OPTIMAL_RANGES,
  KNEE_FLEXION_RANGE,
  evaluateAngle,
  AngleStatus
} from '../utils/AngleCalculator';

/**
 * Ein einzelner Frame mit allen Winkeldaten
 */
export interface FrameData {
  timestamp: number;
  angles: BiomechanicalAngles;
  confidence: number;
}

/**
 * Ein vollständiger Pedalzyklus
 */
export interface PedalCycle {
  startTime: number;
  endTime: number;
  kneeMin: number;
  kneeMax: number;
  frames: FrameData[];
}

/**
 * Statistiken für einen Winkel über mehrere Zyklen
 */
export interface AngleStatistics {
  min: number;
  max: number;
  average: number;
  stdDev: number;
  samples: number;
}

/**
 * Komplette Analyse-Ergebnisse
 */
export interface AnalysisResults {
  cycleCount: number;
  duration: number;
  timestamp: Date;
  statistics: {
    knee: {
      extension: AngleStatistics; // Max-Winkel (gestrecktes Bein)
      flexion: AngleStatistics;   // Min-Winkel (gebeugtes Bein)
    };
    hip: AngleStatistics;
    ankle: AngleStatistics;
    elbow: AngleStatistics;
    back: AngleStatistics;
  };
  evaluations: {
    kneeExtension: AngleStatus;
    kneeFlexion: AngleStatus;
    hip: AngleStatus;
    ankle: AngleStatus;
    elbow: AngleStatus;
    back: AngleStatus;
  };
  rawCycles: PedalCycle[];
}

/**
 * Konfiguration für den Analyzer
 */
export interface AnalyzerConfig {
  targetCycles: number;          // Anzahl gewünschter Zyklen (Standard: 5)
  minCycleFrames: number;        // Minimale Frames pro Zyklus
  minConfidence: number;         // Minimale Keypoint-Confidence
  kneeThresholdHigh: number;     // Schwelle für gestrecktes Bein (°)
  kneeThresholdLow: number;      // Schwelle für gebeugtes Bein (°)
  smoothingWindow: number;       // Frames für Glättung
}

const DEFAULT_CONFIG: AnalyzerConfig = {
  targetCycles: 5,
  minCycleFrames: 10,
  minConfidence: 0.5,
  kneeThresholdHigh: 120,
  kneeThresholdLow: 100,
  smoothingWindow: 5
};

type CyclePhase = 'searching' | 'flexion' | 'extension';

/**
 * BiomechanicalAnalyzer Klasse
 */
export class BiomechanicalAnalyzer {
  private config: AnalyzerConfig;
  private frames: FrameData[] = [];
  private cycles: PedalCycle[] = [];
  private currentPhase: CyclePhase = 'searching';
  private cycleStartIndex: number = 0;
  private smoothedKneeAngles: number[] = [];

  constructor(config: Partial<AnalyzerConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Fügt einen neuen Frame zur Analyse hinzu
   */
  addFrame(angles: BiomechanicalAngles, confidence: number): void {
    const frame: FrameData = {
      timestamp: performance.now(),
      angles,
      confidence
    };

    this.frames.push(frame);

    // Zyklus-Erkennung nur wenn Kniewinkel verfügbar
    if (angles.knee !== null && confidence >= this.config.minConfidence) {
      this.detectCycle(angles.knee);
    }
  }

  /**
   * Erkennt Pedalzyklen basierend auf Kniewinkel-Variation
   */
  private detectCycle(kneeAngle: number): void {
    // Glättung anwenden
    this.smoothedKneeAngles.push(kneeAngle);
    if (this.smoothedKneeAngles.length > this.config.smoothingWindow) {
      this.smoothedKneeAngles.shift();
    }

    const smoothedAngle = this.getSmoothedValue(this.smoothedKneeAngles);

    switch (this.currentPhase) {
      case 'searching':
        // Warte auf gestrecktes Bein (hoher Winkel)
        if (smoothedAngle > this.config.kneeThresholdHigh) {
          this.currentPhase = 'extension';
          this.cycleStartIndex = this.frames.length - 1;
        }
        break;

      case 'extension':
        // Bein war gestreckt, warte auf Beugung
        if (smoothedAngle < this.config.kneeThresholdLow) {
          this.currentPhase = 'flexion';
        }
        break;

      case 'flexion':
        // Bein war gebeugt, warte auf erneute Streckung = Zyklus komplett
        if (smoothedAngle > this.config.kneeThresholdHigh) {
          this.completeCycle();
          this.cycleStartIndex = this.frames.length - 1;
          this.currentPhase = 'extension';
        }
        break;
    }
  }

  /**
   * Schließt einen Zyklus ab
   */
  private completeCycle(): void {
    const cycleFrames = this.frames.slice(this.cycleStartIndex);

    if (cycleFrames.length < this.config.minCycleFrames) {
      return; // Zu kurzer Zyklus, ignorieren
    }

    const kneeAngles = cycleFrames
      .map(f => f.angles.knee)
      .filter((a): a is number => a !== null);

    if (kneeAngles.length === 0) return;

    const cycle: PedalCycle = {
      startTime: cycleFrames[0].timestamp,
      endTime: cycleFrames[cycleFrames.length - 1].timestamp,
      kneeMin: Math.min(...kneeAngles),
      kneeMax: Math.max(...kneeAngles),
      frames: cycleFrames
    };

    this.cycles.push(cycle);
  }

  /**
   * Berechnet geglätteten Wert
   */
  private getSmoothedValue(values: number[]): number {
    if (values.length === 0) return 0;
    return values.reduce((a, b) => a + b, 0) / values.length;
  }

  /**
   * Gibt die Anzahl erkannter Zyklen zurück
   */
  getCycleCount(): number {
    return this.cycles.length;
  }

  /**
   * Gibt den aktuellen Fortschritt zurück (0-1)
   */
  getProgress(): number {
    return Math.min(1, this.cycles.length / this.config.targetCycles);
  }

  /**
   * Prüft ob genug Zyklen erfasst wurden
   */
  isComplete(): boolean {
    return this.cycles.length >= this.config.targetCycles;
  }

  /**
   * Gibt die aktuelle Phase zurück
   */
  getCurrentPhase(): CyclePhase {
    return this.currentPhase;
  }

  /**
   * Berechnet Statistiken für ein Array von Werten
   */
  private calculateStatistics(values: number[]): AngleStatistics {
    if (values.length === 0) {
      return { min: 0, max: 0, average: 0, stdDev: 0, samples: 0 };
    }

    const min = Math.min(...values);
    const max = Math.max(...values);
    const average = values.reduce((a, b) => a + b, 0) / values.length;

    const squaredDiffs = values.map(v => Math.pow(v - average, 2));
    const avgSquaredDiff = squaredDiffs.reduce((a, b) => a + b, 0) / values.length;
    const stdDev = Math.sqrt(avgSquaredDiff);

    return { min, max, average, stdDev, samples: values.length };
  }

  /**
   * Extrahiert alle gültigen Werte eines Winkels aus allen Zyklen
   */
  private extractAngleValues(
    angleKey: keyof BiomechanicalAngles
  ): number[] {
    const values: number[] = [];

    for (const cycle of this.cycles) {
      for (const frame of cycle.frames) {
        const angle = frame.angles[angleKey];
        if (angle !== null && frame.confidence >= this.config.minConfidence) {
          values.push(angle);
        }
      }
    }

    return values;
  }

  /**
   * Generiert die vollständigen Analyse-Ergebnisse
   */
  getResults(): AnalysisResults | null {
    if (this.cycles.length === 0) {
      return null;
    }

    // Knie-Statistiken (Min/Max pro Zyklus)
    const kneeMaxValues = this.cycles.map(c => c.kneeMax);
    const kneeMinValues = this.cycles.map(c => c.kneeMin);

    // Andere Winkel-Statistiken
    const hipValues = this.extractAngleValues('hip');
    const ankleValues = this.extractAngleValues('ankle');
    const elbowValues = this.extractAngleValues('elbow');
    const backValues = this.extractAngleValues('back');

    const kneeExtensionStats = this.calculateStatistics(kneeMaxValues);
    const kneeFlexionStats = this.calculateStatistics(kneeMinValues);

    // Bewertungen berechnen
    const evaluations = {
      kneeExtension: evaluateAngle(kneeExtensionStats.average, OPTIMAL_RANGES.knee),
      kneeFlexion: evaluateAngle(kneeFlexionStats.average, KNEE_FLEXION_RANGE),
      hip: evaluateAngle(
        this.calculateStatistics(hipValues).average,
        OPTIMAL_RANGES.hip
      ),
      ankle: evaluateAngle(
        this.calculateStatistics(ankleValues).average,
        OPTIMAL_RANGES.ankle
      ),
      elbow: evaluateAngle(
        this.calculateStatistics(elbowValues).average,
        OPTIMAL_RANGES.elbow
      ),
      back: evaluateAngle(
        this.calculateStatistics(backValues).average,
        OPTIMAL_RANGES.back
      )
    };

    const firstCycle = this.cycles[0];
    const lastCycle = this.cycles[this.cycles.length - 1];

    return {
      cycleCount: this.cycles.length,
      duration: lastCycle.endTime - firstCycle.startTime,
      timestamp: new Date(),
      statistics: {
        knee: {
          extension: kneeExtensionStats,
          flexion: kneeFlexionStats
        },
        hip: this.calculateStatistics(hipValues),
        ankle: this.calculateStatistics(ankleValues),
        elbow: this.calculateStatistics(elbowValues),
        back: this.calculateStatistics(backValues)
      },
      evaluations,
      rawCycles: this.cycles
    };
  }

  /**
   * Setzt den Analyzer zurück
   */
  reset(): void {
    this.frames = [];
    this.cycles = [];
    this.currentPhase = 'searching';
    this.cycleStartIndex = 0;
    this.smoothedKneeAngles = [];
  }

  /**
   * Aktualisiert die Konfiguration
   */
  updateConfig(config: Partial<AnalyzerConfig>): void {
    this.config = { ...this.config, ...config };
  }
}

/**
 * Singleton-Instanz für globalen Zugriff
 */
let analyzerInstance: BiomechanicalAnalyzer | null = null;

export function getAnalyzer(config?: Partial<AnalyzerConfig>): BiomechanicalAnalyzer {
  if (!analyzerInstance) {
    analyzerInstance = new BiomechanicalAnalyzer(config);
  }
  return analyzerInstance;
}

export function resetAnalyzer(): void {
  if (analyzerInstance) {
    analyzerInstance.reset();
  }
  analyzerInstance = null;
}
