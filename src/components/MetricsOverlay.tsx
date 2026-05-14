import { getAngleColor, isKneeAngleOptimal } from '../utils/AngleCalculator';

interface MetricsOverlayProps {
  inferenceTime: number;
  fps: number;
  frameCount: number;
  kneeAngle: number | null;
  hipAngle: number | null;
}

export function MetricsOverlay({
  inferenceTime,
  fps,
  frameCount,
  kneeAngle,
  hipAngle
}: MetricsOverlayProps) {
  return (
    <div className="bg-gray-800/90 backdrop-blur-sm rounded-lg p-4 shadow-xl border border-gray-700">
      <h2 className="text-lg font-bold mb-4 text-green-400 border-b border-gray-600 pb-2">
        Live Metriken
      </h2>

      {/* Performance Metriken */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <MetricCard
          label="Inferenz"
          value={`${inferenceTime.toFixed(1)} ms`}
          color={inferenceTime < 33 ? '#22c55e' : inferenceTime < 50 ? '#eab308' : '#ef4444'}
        />
        <MetricCard
          label="FPS"
          value={fps.toFixed(1)}
          color={fps >= 25 ? '#22c55e' : fps >= 15 ? '#eab308' : '#ef4444'}
        />
      </div>

      {/* Winkel Metriken */}
      <div className="space-y-4">
        <div className="bg-gray-900/50 rounded-lg p-4">
          <div className="flex justify-between items-center mb-2">
            <span className="text-gray-400 text-sm">Kniewinkel</span>
            {kneeAngle !== null && (
              <span
                className="text-xs px-2 py-1 rounded"
                style={{
                  backgroundColor: isKneeAngleOptimal(kneeAngle) ? '#22c55e20' : '#ef444420',
                  color: isKneeAngleOptimal(kneeAngle) ? '#22c55e' : '#ef4444'
                }}
              >
                {isKneeAngleOptimal(kneeAngle) ? 'Optimal' : 'Anpassen'}
              </span>
            )}
          </div>
          <div className="flex items-baseline gap-1">
            <span
              className="text-3xl font-mono font-bold"
              style={{ color: kneeAngle !== null ? getAngleColor(kneeAngle) : '#6b7280' }}
            >
              {kneeAngle !== null ? kneeAngle.toFixed(1) : '--'}
            </span>
            <span className="text-gray-500">°</span>
          </div>
          <div className="mt-2 text-xs text-gray-500">
            Zielbereich: 140° - 150°
          </div>
          {/* Progress Bar */}
          {kneeAngle !== null && (
            <div className="mt-2 h-2 bg-gray-700 rounded-full overflow-hidden">
              <div
                className="h-full transition-all duration-300"
                style={{
                  width: `${Math.min(100, Math.max(0, ((kneeAngle - 90) / 90) * 100))}%`,
                  backgroundColor: getAngleColor(kneeAngle)
                }}
              />
            </div>
          )}
        </div>

        <div className="bg-gray-900/50 rounded-lg p-4">
          <div className="flex justify-between items-center mb-2">
            <span className="text-gray-400 text-sm">Hüftwinkel</span>
          </div>
          <div className="flex items-baseline gap-1">
            <span className="text-3xl font-mono font-bold text-blue-400">
              {hipAngle !== null ? hipAngle.toFixed(1) : '--'}
            </span>
            <span className="text-gray-500">°</span>
          </div>
          <div className="mt-2 text-xs text-gray-500">
            Oberkörperneigung
          </div>
        </div>
      </div>

      {/* Frame Counter */}
      <div className="mt-4 pt-4 border-t border-gray-700 text-xs text-gray-500 text-center">
        Frames verarbeitet: {frameCount.toLocaleString()}
      </div>
    </div>
  );
}

interface MetricCardProps {
  label: string;
  value: string;
  color: string;
}

function MetricCard({ label, value, color }: MetricCardProps) {
  return (
    <div className="bg-gray-900/50 rounded-lg p-3 text-center">
      <div className="text-gray-400 text-xs mb-1">{label}</div>
      <div className="text-xl font-mono font-bold" style={{ color }}>
        {value}
      </div>
    </div>
  );
}
