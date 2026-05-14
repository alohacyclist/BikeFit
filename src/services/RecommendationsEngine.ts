/**
 * RecommendationsEngine - Generiert konkrete Anpassungsempfehlungen
 *
 * Basierend auf den Analyse-Ergebnissen werden priorisierte
 * Empfehlungen für die Fahrradeinstellung generiert.
 */

import { AnalysisResults } from './BiomechanicalAnalyzer';
import {
  OPTIMAL_RANGES,
  KNEE_FLEXION_RANGE,
  AngleStatus
} from '../utils/AngleCalculator';

/**
 * Eine einzelne Empfehlung
 */
export interface Recommendation {
  id: string;
  priority: 'critical' | 'recommended' | 'optional';
  category: 'saddle' | 'handlebar' | 'cleat' | 'general';
  title: string;
  description: string;
  adjustment: string;
  reason: string;
  expectedImprovement: string;
  icon: string;
}

/**
 * Vollständiger Empfehlungs-Report
 */
export interface RecommendationReport {
  timestamp: Date;
  overallScore: number; // 0-100
  overallStatus: AngleStatus;
  recommendations: Recommendation[];
  summary: string;
}

/**
 * Generiert Empfehlungen basierend auf Analyse-Ergebnissen
 */
export function generateRecommendations(
  results: AnalysisResults
): RecommendationReport {
  const recommendations: Recommendation[] = [];

  // Knie-Streckung analysieren
  analyzeKneeExtension(results, recommendations);

  // Knie-Beugung analysieren
  analyzeKneeFlexion(results, recommendations);

  // Hüftwinkel analysieren
  analyzeHipAngle(results, recommendations);

  // Rückenwinkel analysieren
  analyzeBackAngle(results, recommendations);

  // Ellenbogenwinkel analysieren
  analyzeElbowAngle(results, recommendations);

  // Knöchelwinkel analysieren
  analyzeAnkleAngle(results, recommendations);

  // Nach Priorität sortieren
  recommendations.sort((a, b) => {
    const priorityOrder = { critical: 0, recommended: 1, optional: 2 };
    return priorityOrder[a.priority] - priorityOrder[b.priority];
  });

  // Gesamtbewertung berechnen
  const overallScore = calculateOverallScore(results);
  const overallStatus = getOverallStatus(overallScore);

  return {
    timestamp: new Date(),
    overallScore,
    overallStatus,
    recommendations,
    summary: generateSummary(overallScore, recommendations.length)
  };
}

/**
 * Analysiert die Kniestreckung (unterer Totpunkt)
 */
function analyzeKneeExtension(
  results: AnalysisResults,
  recommendations: Recommendation[]
): void {
  const stats = results.statistics.knee.extension;
  const status = results.evaluations.kneeExtension;
  const optimal = OPTIMAL_RANGES.knee;

  if (status === 'optimal') return;

  const diff = stats.average - optimal.min;

  if (stats.average < optimal.min) {
    // Knie zu stark gebeugt bei Streckung → Sattel höher
    const adjustment = Math.abs(diff) * 0.5; // Ca. 0.5mm pro Grad
    recommendations.push({
      id: 'knee-ext-low',
      priority: status === 'critical' ? 'critical' : 'recommended',
      category: 'saddle',
      title: 'Sattel höher stellen',
      description: `Kniewinkel bei Streckung (${stats.average.toFixed(1)}°) ist unter dem Optimum.`,
      adjustment: `Sattel ${adjustment.toFixed(0)}mm nach oben`,
      reason: `Optimaler Bereich: ${optimal.min}°-${optimal.max}°. Aktuell: ${stats.average.toFixed(1)}°`,
      expectedImprovement: `Verbessert Kniestreckung um ca. ${Math.abs(diff).toFixed(0)}°`,
      icon: '↑'
    });
  } else if (stats.average > optimal.max) {
    // Knie zu stark gestreckt → Sattel niedriger
    const adjustment = Math.abs(diff) * 0.5;
    recommendations.push({
      id: 'knee-ext-high',
      priority: status === 'critical' ? 'critical' : 'recommended',
      category: 'saddle',
      title: 'Sattel niedriger stellen',
      description: `Kniewinkel bei Streckung (${stats.average.toFixed(1)}°) ist über dem Optimum.`,
      adjustment: `Sattel ${adjustment.toFixed(0)}mm nach unten`,
      reason: `Optimaler Bereich: ${optimal.min}°-${optimal.max}°. Aktuell: ${stats.average.toFixed(1)}°`,
      expectedImprovement: `Reduziert Überstreckung um ca. ${Math.abs(diff).toFixed(0)}°`,
      icon: '↓'
    });
  }
}

/**
 * Analysiert die Kniebeugung (oberer Totpunkt)
 */
function analyzeKneeFlexion(
  results: AnalysisResults,
  recommendations: Recommendation[]
): void {
  const stats = results.statistics.knee.flexion;
  const status = results.evaluations.kneeFlexion;
  const optimal = KNEE_FLEXION_RANGE;

  if (status === 'optimal') return;

  if (stats.average < optimal.min) {
    // Zu starke Beugung am oberen Totpunkt
    recommendations.push({
      id: 'knee-flex-low',
      priority: status === 'critical' ? 'critical' : 'recommended',
      category: 'saddle',
      title: 'Sattel nach hinten',
      description: `Kniebeugung am oberen Totpunkt (${stats.average.toFixed(1)}°) ist zu stark.`,
      adjustment: `Sattel 5-10mm nach hinten verschieben`,
      reason: `Optimaler Bereich: ${optimal.min}°-${optimal.max}°. Aktuell: ${stats.average.toFixed(1)}°`,
      expectedImprovement: `Reduziert Kniebelastung am oberen Totpunkt`,
      icon: '←'
    });
  } else if (stats.average > optimal.max) {
    // Zu wenig Beugung
    recommendations.push({
      id: 'knee-flex-high',
      priority: status === 'critical' ? 'critical' : 'recommended',
      category: 'saddle',
      title: 'Sattel nach vorne',
      description: `Kniebeugung am oberen Totpunkt (${stats.average.toFixed(1)}°) ist zu gering.`,
      adjustment: `Sattel 5-10mm nach vorne verschieben`,
      reason: `Optimaler Bereich: ${optimal.min}°-${optimal.max}°. Aktuell: ${stats.average.toFixed(1)}°`,
      expectedImprovement: `Verbessert Kraftübertragung im oberen Bereich`,
      icon: '→'
    });
  }
}

/**
 * Analysiert den Hüftwinkel
 */
function analyzeHipAngle(
  results: AnalysisResults,
  recommendations: Recommendation[]
): void {
  const stats = results.statistics.hip;
  const status = results.evaluations.hip;
  const optimal = OPTIMAL_RANGES.hip;

  if (status === 'optimal' || stats.samples === 0) return;

  if (stats.average < optimal.min) {
    // Hüfte zu geschlossen
    recommendations.push({
      id: 'hip-closed',
      priority: status === 'critical' ? 'critical' : 'recommended',
      category: 'saddle',
      title: 'Hüftöffnung verbessern',
      description: `Hüftwinkel (${stats.average.toFixed(1)}°) ist zu geschlossen.`,
      adjustment: `Sattel 5-15mm nach hinten und/oder Vorbau länger`,
      reason: `Optimaler Bereich: ${optimal.min}°-${optimal.max}°. Aktuell: ${stats.average.toFixed(1)}°`,
      expectedImprovement: `Bessere Atmung und Kraftentfaltung`,
      icon: '↔'
    });
  } else if (stats.average > optimal.max) {
    // Hüfte zu offen
    recommendations.push({
      id: 'hip-open',
      priority: 'optional',
      category: 'saddle',
      title: 'Hüftwinkel anpassen',
      description: `Hüftwinkel (${stats.average.toFixed(1)}°) ist sehr offen.`,
      adjustment: `Sattel leicht nach vorne oder Vorbau kürzer`,
      reason: `Optimaler Bereich: ${optimal.min}°-${optimal.max}°. Aktuell: ${stats.average.toFixed(1)}°`,
      expectedImprovement: `Aerodynamischere Position möglich`,
      icon: '↔'
    });
  }
}

/**
 * Analysiert den Rückenwinkel
 */
function analyzeBackAngle(
  results: AnalysisResults,
  recommendations: Recommendation[]
): void {
  const stats = results.statistics.back;
  const status = results.evaluations.back;
  const optimal = OPTIMAL_RANGES.back;

  if (status === 'optimal' || stats.samples === 0) return;

  if (stats.average < optimal.min) {
    // Rücken zu aufrecht
    recommendations.push({
      id: 'back-upright',
      priority: 'optional',
      category: 'handlebar',
      title: 'Aerodynamischere Position',
      description: `Rückenneigung (${stats.average.toFixed(1)}°) ist relativ aufrecht.`,
      adjustment: `Vorbau 10-20mm länger oder Spacer entfernen`,
      reason: `Optimaler Bereich: ${optimal.min}°-${optimal.max}°. Aktuell: ${stats.average.toFixed(1)}°`,
      expectedImprovement: `Verbesserte Aerodynamik bei Beibehaltung des Komforts`,
      icon: '↙'
    });
  } else if (stats.average > optimal.max) {
    // Rücken zu gestreckt
    recommendations.push({
      id: 'back-stretched',
      priority: status === 'critical' ? 'critical' : 'recommended',
      category: 'handlebar',
      title: 'Komfortablere Position',
      description: `Rückenneigung (${stats.average.toFixed(1)}°) ist sehr gestreckt.`,
      adjustment: `Vorbau 10-20mm kürzer oder Spacer hinzufügen`,
      reason: `Optimaler Bereich: ${optimal.min}°-${optimal.max}°. Aktuell: ${stats.average.toFixed(1)}°`,
      expectedImprovement: `Reduziert Nacken- und Rückenbelastung`,
      icon: '↗'
    });
  }
}

/**
 * Analysiert den Ellenbogenwinkel
 */
function analyzeElbowAngle(
  results: AnalysisResults,
  recommendations: Recommendation[]
): void {
  const stats = results.statistics.elbow;
  const status = results.evaluations.elbow;
  const optimal = OPTIMAL_RANGES.elbow;

  if (status === 'optimal' || stats.samples === 0) return;

  if (stats.average < optimal.min) {
    // Arme zu stark gebeugt
    recommendations.push({
      id: 'elbow-bent',
      priority: 'optional',
      category: 'handlebar',
      title: 'Armposition optimieren',
      description: `Ellenbogenwinkel (${stats.average.toFixed(1)}°) zeigt stark gebeugte Arme.`,
      adjustment: `Lenker 10-15mm weiter weg positionieren`,
      reason: `Optimaler Bereich: ${optimal.min}°-${optimal.max}°. Aktuell: ${stats.average.toFixed(1)}°`,
      expectedImprovement: `Bessere Stoßdämpfung und Kontrolle`,
      icon: '↔'
    });
  } else if (stats.average > optimal.max) {
    // Arme zu gestreckt
    recommendations.push({
      id: 'elbow-straight',
      priority: status === 'critical' ? 'recommended' : 'optional',
      category: 'handlebar',
      title: 'Leichte Armbeugung',
      description: `Ellenbogenwinkel (${stats.average.toFixed(1)}°) zeigt fast gestreckte Arme.`,
      adjustment: `Lenker 10-15mm näher positionieren`,
      reason: `Optimaler Bereich: ${optimal.min}°-${optimal.max}°. Aktuell: ${stats.average.toFixed(1)}°`,
      expectedImprovement: `Bessere Stoßabsorption und reduzierte Handgelenkbelastung`,
      icon: '↔'
    });
  }
}

/**
 * Analysiert den Knöchelwinkel
 */
function analyzeAnkleAngle(
  results: AnalysisResults,
  recommendations: Recommendation[]
): void {
  const stats = results.statistics.ankle;
  const status = results.evaluations.ankle;
  const optimal = OPTIMAL_RANGES.ankle;

  if (status === 'optimal' || stats.samples === 0) return;

  if (stats.average < optimal.min) {
    // Fuß zu stark angewinkelt
    recommendations.push({
      id: 'ankle-dorsi',
      priority: 'optional',
      category: 'cleat',
      title: 'Cleat-Position prüfen',
      description: `Knöchelwinkel (${stats.average.toFixed(1)}°) zeigt starke Dorsalflexion.`,
      adjustment: `Cleats 2-3mm nach hinten oder Ferse leicht senken`,
      reason: `Optimaler Bereich: ${optimal.min}°-${optimal.max}°. Aktuell: ${stats.average.toFixed(1)}°`,
      expectedImprovement: `Gleichmäßigere Kraftübertragung`,
      icon: '⟲'
    });
  } else if (stats.average > optimal.max) {
    // Fuß zu flach/gestreckt
    recommendations.push({
      id: 'ankle-plantar',
      priority: 'optional',
      category: 'cleat',
      title: 'Fußposition anpassen',
      description: `Knöchelwinkel (${stats.average.toFixed(1)}°) zeigt Plantarflexion.`,
      adjustment: `Cleats 2-3mm nach vorne oder bewusst Ferse heben`,
      reason: `Optimaler Bereich: ${optimal.min}°-${optimal.max}°. Aktuell: ${stats.average.toFixed(1)}°`,
      expectedImprovement: `Bessere Wadenaktivierung beim Pedalieren`,
      icon: '⟲'
    });
  }
}

/**
 * Berechnet eine Gesamtbewertung (0-100)
 */
function calculateOverallScore(results: AnalysisResults): number {
  const evaluations = results.evaluations;
  let score = 100;

  // Gewichtung: Knie am wichtigsten
  const weights = {
    kneeExtension: 25,
    kneeFlexion: 20,
    hip: 20,
    back: 15,
    elbow: 10,
    ankle: 10
  };

  const penalties = {
    optimal: 0,
    acceptable: 0.5,
    critical: 1
  };

  for (const [key, weight] of Object.entries(weights)) {
    const status = evaluations[key as keyof typeof evaluations];
    score -= weight * penalties[status];
  }

  return Math.max(0, Math.round(score));
}

/**
 * Bestimmt den Gesamtstatus basierend auf dem Score
 */
function getOverallStatus(score: number): AngleStatus {
  if (score >= 80) return 'optimal';
  if (score >= 60) return 'acceptable';
  return 'critical';
}

/**
 * Generiert eine Zusammenfassung
 */
function generateSummary(score: number, recommendationCount: number): string {
  if (score >= 90) {
    return 'Ausgezeichnete Fahrradeinstellung! Nur minimale Optimierungen möglich.';
  } else if (score >= 80) {
    return 'Gute Einstellung mit kleinem Optimierungspotential.';
  } else if (score >= 70) {
    return 'Solide Grundeinstellung. Einige Anpassungen empfohlen.';
  } else if (score >= 60) {
    return 'Verbesserungspotential vorhanden. Bitte Empfehlungen beachten.';
  } else {
    return `${recommendationCount} wichtige Anpassungen erforderlich für optimale Ergonomie.`;
  }
}
