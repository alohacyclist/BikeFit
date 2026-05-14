import { QuantizationLevel } from '../types/quantization';

interface QuantizationControlsProps {
  currentLevel: QuantizationLevel;
  onLevelChange: (level: QuantizationLevel) => void;
  isLoading: boolean;
  isWarmingUp: boolean;
  participantId: string;
  onParticipantIdChange: (id: string) => void;
  onExport: () => void;
}

const LEVELS: QuantizationLevel[] = ['fp32', 'fp16', 'int8'];

export function QuantizationControls({
  currentLevel,
  onLevelChange,
  isLoading,
  isWarmingUp,
  participantId,
  onParticipantIdChange,
  onExport,
}: QuantizationControlsProps) {
  const disabled = isLoading || isWarmingUp;

  return (
    <div className="bg-gray-800/60 rounded-lg p-4 border border-gray-700 space-y-3">
      <div>
        <h3 className="text-sm font-semibold text-gray-300 mb-2">
          Quantisierungsstufe
        </h3>
        <div className="grid grid-cols-3 gap-2">
          {LEVELS.map((level) => {
            const active = currentLevel === level;
            return (
              <button
                key={level}
                onClick={() => onLevelChange(level)}
                disabled={disabled}
                className={`py-2 px-3 rounded-md text-sm font-mono uppercase transition-colors ${
                  active
                    ? 'bg-blue-600 text-white border border-blue-400'
                    : 'bg-gray-700 text-gray-300 border border-gray-600 hover:bg-gray-600'
                } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                {level}
              </button>
            );
          })}
        </div>
      </div>

      <div>
        <label className="block text-xs text-gray-400 mb-1">
          Teilnehmer-ID
        </label>
        <input
          type="text"
          value={participantId}
          onChange={(e) => onParticipantIdChange(e.target.value)}
          placeholder="P01"
          className="w-full px-3 py-2 rounded-md bg-gray-900 border border-gray-700 text-sm text-gray-200 font-mono focus:outline-none focus:border-blue-500"
        />
      </div>

      {/* Warmup-Indikator */}
      {isWarmingUp && (
        <div className="flex items-center gap-2 bg-yellow-900/40 border border-yellow-700/50 rounded-md px-3 py-2">
          <span className="w-2 h-2 rounded-full bg-yellow-400 animate-pulse" />
          <span className="text-xs text-yellow-300">
            Aufwärmen... (30 Frames Warmup)
          </span>
        </div>
      )}

      {isLoading && !isWarmingUp && (
        <div className="text-xs text-gray-400">Modell wird geladen...</div>
      )}

      {/* Export-Button — nur aktiv wenn nicht im Warmup */}
      <button
        onClick={onExport}
        disabled={isWarmingUp}
        className={`w-full py-2 px-3 rounded-md text-sm font-semibold transition-colors ${
          isWarmingUp
            ? 'bg-gray-700 text-gray-500 cursor-not-allowed'
            : 'bg-green-600 hover:bg-green-700 text-white'
        }`}
      >
        JSON exportieren
      </button>
    </div>
  );
}
