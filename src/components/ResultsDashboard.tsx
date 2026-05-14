import { useMemo } from 'react';
import { AnalysisResults } from '../services/BiomechanicalAnalyzer';
import {
  generateRecommendations,
  Recommendation
} from '../services/RecommendationsEngine';
import {
  OPTIMAL_RANGES,
  KNEE_FLEXION_RANGE,
  getStatusColor,
  AngleStatus
} from '../utils/AngleCalculator';

interface ResultsDashboardProps {
  results: AnalysisResults;
  onNewAnalysis: () => void;
  onShowHistory: () => void;
}

export function ResultsDashboard({
  results,
  onNewAnalysis,
  onShowHistory
}: ResultsDashboardProps) {
  const report = useMemo(() => generateRecommendations(results), [results]);

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      {/* Header mit Score */}
      <div className="bg-gray-800/50 rounded-xl p-6 border border-gray-700">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <h2 className="text-2xl font-bold text-white mb-2">
              Analyse abgeschlossen
            </h2>
            <p className="text-gray-400">
              {results.cycleCount} Pedalzyklen in{' '}
              {(results.duration / 1000).toFixed(1)}s analysiert
            </p>
          </div>
          <div className="text-center">
            <div
              className="text-5xl font-bold"
              style={{ color: getStatusColor(report.overallStatus) }}
            >
              {report.overallScore}
            </div>
            <div className="text-sm text-gray-400 mt-1">Gesamtbewertung</div>
          </div>
        </div>
        <div className="mt-4 p-4 rounded-lg" style={{
          backgroundColor: `${getStatusColor(report.overallStatus)}20`
        }}>
          <p style={{ color: getStatusColor(report.overallStatus) }}>
            {report.summary}
          </p>
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        {/* Gemessene Winkel */}
        <div className="bg-gray-800/50 rounded-xl p-6 border border-gray-700">
          <h3 className="text-lg font-semibold text-green-400 mb-4 flex items-center gap-2">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            Gemessene Winkel
          </h3>

          <div className="space-y-4">
            {/* Kniewinkel */}
            <AngleResultCard
              title="Kniewinkel (Streckung)"
              value={results.statistics.knee.extension.average}
              min={results.statistics.knee.extension.min}
              max={results.statistics.knee.extension.max}
              optimalRange={OPTIMAL_RANGES.knee}
              status={results.evaluations.kneeExtension}
            />

            <AngleResultCard
              title="Kniewinkel (Beugung)"
              value={results.statistics.knee.flexion.average}
              min={results.statistics.knee.flexion.min}
              max={results.statistics.knee.flexion.max}
              optimalRange={KNEE_FLEXION_RANGE}
              status={results.evaluations.kneeFlexion}
            />

            <AngleResultCard
              title="Hüftwinkel"
              value={results.statistics.hip.average}
              stdDev={results.statistics.hip.stdDev}
              optimalRange={OPTIMAL_RANGES.hip}
              status={results.evaluations.hip}
            />

            <AngleResultCard
              title="Rückenneigung"
              value={results.statistics.back.average}
              stdDev={results.statistics.back.stdDev}
              optimalRange={OPTIMAL_RANGES.back}
              status={results.evaluations.back}
            />

            <AngleResultCard
              title="Ellenbogenwinkel"
              value={results.statistics.elbow.average}
              stdDev={results.statistics.elbow.stdDev}
              optimalRange={OPTIMAL_RANGES.elbow}
              status={results.evaluations.elbow}
            />

            <AngleResultCard
              title="Knöchelwinkel"
              value={results.statistics.ankle.average}
              stdDev={results.statistics.ankle.stdDev}
              optimalRange={OPTIMAL_RANGES.ankle}
              status={results.evaluations.ankle}
            />
          </div>
        </div>

        {/* Empfehlungen */}
        <div className="bg-gray-800/50 rounded-xl p-6 border border-gray-700">
          <h3 className="text-lg font-semibold text-green-400 mb-4 flex items-center gap-2">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
            Empfehlungen
            {report.recommendations.length > 0 && (
              <span className="ml-auto text-sm text-gray-400">
                {report.recommendations.length} Anpassungen
              </span>
            )}
          </h3>

          {report.recommendations.length === 0 ? (
            <div className="text-center py-8 text-gray-400">
              <svg className="w-16 h-16 mx-auto mb-4 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <p className="text-lg font-medium text-green-400">Optimale Einstellung!</p>
              <p className="text-sm mt-2">Keine Anpassungen erforderlich.</p>
            </div>
          ) : (
            <div className="space-y-3">
              {report.recommendations.map((rec, index) => (
                <RecommendationCard key={rec.id} recommendation={rec} index={index + 1} />
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Aktionsbuttons */}
      <div className="flex flex-col sm:flex-row gap-4 justify-center">
        <button
          onClick={onNewAnalysis}
          className="py-3 px-8 rounded-lg font-semibold bg-green-500 hover:bg-green-600 text-white transition-all flex items-center justify-center gap-2"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          Neue Messung
        </button>
        <button
          onClick={onShowHistory}
          className="py-3 px-8 rounded-lg font-semibold bg-gray-700 hover:bg-gray-600 text-gray-300 transition-all flex items-center justify-center gap-2"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          Verlauf anzeigen
        </button>
      </div>
    </div>
  );
}

interface AngleResultCardProps {
  title: string;
  value: number;
  min?: number;
  max?: number;
  stdDev?: number;
  optimalRange: { min: number; max: number };
  status: AngleStatus;
}

function AngleResultCard({
  title,
  value,
  min,
  max,
  stdDev,
  optimalRange,
  status
}: AngleResultCardProps) {
  const statusIcon = status === 'optimal' ? '✓' : status === 'acceptable' ? '⚠' : '✗';

  return (
    <div className="bg-gray-900/50 rounded-lg p-4">
      <div className="flex justify-between items-start mb-2">
        <span className="text-gray-300 font-medium">{title}</span>
        <span
          className="text-xl"
          style={{ color: getStatusColor(status) }}
        >
          {statusIcon}
        </span>
      </div>
      <div className="flex items-baseline gap-2">
        <span
          className="text-2xl font-mono font-bold"
          style={{ color: getStatusColor(status) }}
        >
          {value.toFixed(1)}°
        </span>
        {min !== undefined && max !== undefined && (
          <span className="text-sm text-gray-500">
            ({min.toFixed(0)}° - {max.toFixed(0)}°)
          </span>
        )}
        {stdDev !== undefined && stdDev > 0 && (
          <span className="text-sm text-gray-500">
            ±{stdDev.toFixed(1)}°
          </span>
        )}
      </div>
      <div className="text-xs text-gray-500 mt-1">
        Optimal: {optimalRange.min}° - {optimalRange.max}°
      </div>
    </div>
  );
}

interface RecommendationCardProps {
  recommendation: Recommendation;
  index: number;
}

function RecommendationCard({ recommendation, index }: RecommendationCardProps) {
  const priorityColors = {
    critical: { bg: 'bg-red-900/30', border: 'border-red-500/50', text: 'text-red-400' },
    recommended: { bg: 'bg-yellow-900/30', border: 'border-yellow-500/50', text: 'text-yellow-400' },
    optional: { bg: 'bg-blue-900/30', border: 'border-blue-500/50', text: 'text-blue-400' }
  };

  const colors = priorityColors[recommendation.priority];
  const priorityLabel = recommendation.priority === 'critical' ? 'Kritisch' :
                       recommendation.priority === 'recommended' ? 'Empfohlen' : 'Optional';

  return (
    <div className={`${colors.bg} rounded-lg p-4 border ${colors.border}`}>
      <div className="flex items-start gap-3">
        <span className={`text-lg font-bold ${colors.text}`}>
          {index}.
        </span>
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-1">
            <span className="font-medium text-white">{recommendation.title}</span>
            <span className={`text-xs px-2 py-0.5 rounded ${colors.text} ${colors.bg}`}>
              {priorityLabel}
            </span>
          </div>
          <p className="text-sm text-gray-400 mb-2">
            {recommendation.description}
          </p>
          <div className="flex items-center gap-2 text-sm">
            <span className={`font-mono font-bold ${colors.text}`}>
              {recommendation.icon} {recommendation.adjustment}
            </span>
          </div>
          <p className="text-xs text-gray-500 mt-2">
            {recommendation.expectedImprovement}
          </p>
        </div>
      </div>
    </div>
  );
}
