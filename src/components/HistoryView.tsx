import { useState, useEffect } from 'react';
import {
  loadHistory,
  deleteFromHistory,
  clearHistory,
  formatDate,
  HistoryEntry
} from '../services/HistoryStorage';
import { getStatusColor } from '../utils/AngleCalculator';

interface HistoryViewProps {
  onSelectEntry: (entry: HistoryEntry) => void;
  onBack: () => void;
  onNewAnalysis: () => void;
}

export function HistoryView({
  onSelectEntry,
  onBack,
  onNewAnalysis
}: HistoryViewProps) {
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [confirmClear, setConfirmClear] = useState(false);

  useEffect(() => {
    setHistory(loadHistory());
  }, []);

  const handleDelete = (id: string) => {
    deleteFromHistory(id);
    setHistory(loadHistory());
  };

  const handleClearAll = () => {
    if (confirmClear) {
      clearHistory();
      setHistory([]);
      setConfirmClear(false);
    } else {
      setConfirmClear(true);
      setTimeout(() => setConfirmClear(false), 3000);
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white">Messverlauf</h2>
          <p className="text-gray-400 text-sm mt-1">
            {history.length} {history.length === 1 ? 'Messung' : 'Messungen'} gespeichert
          </p>
        </div>
        <div className="flex gap-3">
          <button
            onClick={onBack}
            className="py-2 px-4 rounded-lg font-medium bg-gray-700 hover:bg-gray-600 text-gray-300 transition-all"
          >
            Zurück
          </button>
          {history.length > 0 && (
            <button
              onClick={handleClearAll}
              className={`py-2 px-4 rounded-lg font-medium transition-all ${
                confirmClear
                  ? 'bg-red-500 hover:bg-red-600 text-white'
                  : 'bg-gray-700 hover:bg-gray-600 text-gray-300'
              }`}
            >
              {confirmClear ? 'Bestätigen?' : 'Alle löschen'}
            </button>
          )}
        </div>
      </div>

      {/* Historie-Liste */}
      {history.length === 0 ? (
        <div className="bg-gray-800/50 rounded-xl p-12 border border-gray-700 text-center">
          <svg
            className="w-20 h-20 mx-auto mb-4 text-gray-600"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
          <h3 className="text-xl font-semibold text-gray-400 mb-2">
            Noch keine Messungen
          </h3>
          <p className="text-gray-500 mb-6">
            Starte eine neue Analyse, um deinen Fortschritt zu verfolgen.
          </p>
          <button
            onClick={onNewAnalysis}
            className="py-3 px-6 rounded-lg font-semibold bg-green-500 hover:bg-green-600 text-white transition-all"
          >
            Erste Messung starten
          </button>
        </div>
      ) : (
        <div className="space-y-3">
          {history.map((entry) => (
            <HistoryCard
              key={entry.id}
              entry={entry}
              onSelect={() => onSelectEntry(entry)}
              onDelete={() => handleDelete(entry.id)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

interface HistoryCardProps {
  entry: HistoryEntry;
  onSelect: () => void;
  onDelete: () => void;
}

function HistoryCard({ entry, onSelect, onDelete }: HistoryCardProps) {
  const score = entry.recommendations.overallScore;
  const status = entry.recommendations.overallStatus;
  const cycleCount = entry.results.cycleCount;

  return (
    <div
      className="bg-gray-800/50 rounded-xl p-5 border border-gray-700 hover:border-gray-600 transition-all cursor-pointer"
      onClick={onSelect}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          {/* Score Circle */}
          <div
            className="w-14 h-14 rounded-full flex items-center justify-center text-xl font-bold"
            style={{
              backgroundColor: `${getStatusColor(status)}20`,
              color: getStatusColor(status)
            }}
          >
            {score}
          </div>

          {/* Info */}
          <div>
            <div className="font-medium text-white">
              {formatDate(entry.timestamp)}
            </div>
            <div className="text-sm text-gray-400 mt-1">
              {cycleCount} Zyklen analysiert
            </div>
          </div>
        </div>

        {/* Kurzübersicht Winkel */}
        <div className="hidden md:flex items-center gap-4">
          <AngleBadge
            label="Knie"
            value={entry.results.statistics.knee.extension.average}
            status={entry.results.evaluations.kneeExtension}
          />
          <AngleBadge
            label="Hüfte"
            value={entry.results.statistics.hip.average}
            status={entry.results.evaluations.hip}
          />
          <AngleBadge
            label="Rücken"
            value={entry.results.statistics.back.average}
            status={entry.results.evaluations.back}
          />
        </div>

        {/* Aktionen */}
        <div className="flex items-center gap-2">
          <button
            onClick={(e) => {
              e.stopPropagation();
              onDelete();
            }}
            className="p-2 rounded-lg text-gray-500 hover:text-red-400 hover:bg-gray-700 transition-all"
            title="Löschen"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
              />
            </svg>
          </button>
          <svg
            className="w-5 h-5 text-gray-500"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 5l7 7-7 7"
            />
          </svg>
        </div>
      </div>

      {/* Empfehlungen Preview */}
      {entry.recommendations.recommendations.length > 0 && (
        <div className="mt-4 pt-4 border-t border-gray-700">
          <div className="text-xs text-gray-500 mb-2">
            {entry.recommendations.recommendations.length} Empfehlung(en)
          </div>
          <div className="flex flex-wrap gap-2">
            {entry.recommendations.recommendations.slice(0, 3).map((rec) => (
              <span
                key={rec.id}
                className={`text-xs px-2 py-1 rounded ${
                  rec.priority === 'critical'
                    ? 'bg-red-900/30 text-red-400'
                    : rec.priority === 'recommended'
                    ? 'bg-yellow-900/30 text-yellow-400'
                    : 'bg-blue-900/30 text-blue-400'
                }`}
              >
                {rec.title}
              </span>
            ))}
            {entry.recommendations.recommendations.length > 3 && (
              <span className="text-xs text-gray-500">
                +{entry.recommendations.recommendations.length - 3} weitere
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

interface AngleBadgeProps {
  label: string;
  value: number;
  status: 'optimal' | 'acceptable' | 'critical';
}

function AngleBadge({ label, value, status }: AngleBadgeProps) {
  return (
    <div className="text-center">
      <div className="text-xs text-gray-500">{label}</div>
      <div
        className="font-mono font-bold"
        style={{ color: getStatusColor(status) }}
      >
        {value.toFixed(0)}°
      </div>
    </div>
  );
}
