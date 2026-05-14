/**
 * HistoryStorage - LocalStorage-basierte Speicherung von Analyse-Ergebnissen
 */

import { AnalysisResults } from './BiomechanicalAnalyzer';
import { RecommendationReport } from './RecommendationsEngine';

const STORAGE_KEY = 'edgefit_history';
const MAX_HISTORY_ITEMS = 20;

/**
 * Ein gespeicherter Verlaufseintrag
 */
export interface HistoryEntry {
  id: string;
  timestamp: string;
  results: AnalysisResults;
  recommendations: RecommendationReport;
}

/**
 * Serialisierbares Format für AnalysisResults
 */
interface SerializableResults {
  cycleCount: number;
  duration: number;
  timestamp: string;
  statistics: AnalysisResults['statistics'];
  evaluations: AnalysisResults['evaluations'];
}

/**
 * Generiert eine eindeutige ID
 */
function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Konvertiert AnalysisResults für die Speicherung
 */
function serializeResults(results: AnalysisResults): SerializableResults {
  return {
    cycleCount: results.cycleCount,
    duration: results.duration,
    timestamp: results.timestamp.toISOString(),
    statistics: results.statistics,
    evaluations: results.evaluations
    // rawCycles werden nicht gespeichert (zu groß)
  };
}

/**
 * Lädt den Verlauf aus dem LocalStorage
 */
export function loadHistory(): HistoryEntry[] {
  try {
    const data = localStorage.getItem(STORAGE_KEY);
    if (!data) return [];

    const parsed = JSON.parse(data);
    return Array.isArray(parsed) ? parsed : [];
  } catch (error) {
    console.error('Fehler beim Laden des Verlaufs:', error);
    return [];
  }
}

/**
 * Speichert einen neuen Eintrag im Verlauf
 */
export function saveToHistory(
  results: AnalysisResults,
  recommendations: RecommendationReport
): HistoryEntry {
  const entry: HistoryEntry = {
    id: generateId(),
    timestamp: new Date().toISOString(),
    results: {
      ...serializeResults(results),
      timestamp: results.timestamp,
      rawCycles: []
    },
    recommendations: {
      ...recommendations,
      timestamp: recommendations.timestamp
    }
  };

  const history = loadHistory();
  history.unshift(entry);

  // Auf maximale Anzahl begrenzen
  const trimmed = history.slice(0, MAX_HISTORY_ITEMS);

  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(trimmed));
  } catch (error) {
    console.error('Fehler beim Speichern des Verlaufs:', error);
  }

  return entry;
}

/**
 * Löscht einen Eintrag aus dem Verlauf
 */
export function deleteFromHistory(id: string): void {
  const history = loadHistory();
  const filtered = history.filter(entry => entry.id !== id);

  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(filtered));
  } catch (error) {
    console.error('Fehler beim Löschen aus dem Verlauf:', error);
  }
}

/**
 * Löscht den gesamten Verlauf
 */
export function clearHistory(): void {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch (error) {
    console.error('Fehler beim Löschen des Verlaufs:', error);
  }
}

/**
 * Lädt einen einzelnen Eintrag
 */
export function getHistoryEntry(id: string): HistoryEntry | null {
  const history = loadHistory();
  return history.find(entry => entry.id === id) || null;
}

/**
 * Formatiert ein Datum für die Anzeige
 */
export function formatDate(isoString: string): string {
  const date = new Date(isoString);
  return date.toLocaleDateString('de-DE', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  });
}
