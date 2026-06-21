const TARGET_FPS_OPTIONS = [24, 25, 30, 60] as const;

interface VideoSourcePanelProps {
  file: File | null;
  onFileChange: (file: File | null) => void;
  targetFps: number;
  onTargetFpsChange: (fps: number) => void;
  /** Während laufender Aufnahme gesperrt (Quelle/fps dürfen nicht wechseln). */
  disabled?: boolean;
}

/**
 * UI-Panel für die Replay-Datei (CFR) und die Ziel-Framerate.
 * Das Benchmark arbeitet ausschließlich mit vorab aufgenommenen, auf CFR
 * konvertierten Videos — so ist der Frame-Satz über alle
 * Quantisierungsstufen identisch und reproduzierbar.
 */
export function VideoSourcePanel({
  file,
  onFileChange,
  targetFps,
  onTargetFpsChange,
  disabled = false,
}: VideoSourcePanelProps) {
  const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    onFileChange(e.target.files?.[0] ?? null);
  };

  return (
    <div className="bg-gray-800/60 rounded-lg p-4 border border-gray-700 space-y-3">
      <h3 className="text-sm font-semibold text-gray-300">Video-Quelle</h3>

      <div className="space-y-2">
        <label className="block text-xs text-gray-400">
          Replay-Video laden (.mp4 / .webm, CFR erforderlich):
        </label>
        <input
          type="file"
          accept="video/mp4,video/webm,video/*"
          onChange={handleFile}
          disabled={disabled}
          className="block w-full text-xs text-gray-300 file:mr-2 file:py-1 file:px-3 file:rounded-md file:border-0 file:bg-blue-600 file:text-white hover:file:bg-blue-700 disabled:opacity-50"
        />
        {file && (
          <div className="text-xs text-gray-400">
            Geladen: <span className="text-gray-200">{file.name}</span> (
            {(file.size / 1024 / 1024).toFixed(1)} MB)
          </div>
        )}
      </div>

      <div className="space-y-1">
        <label className="block text-xs text-gray-400">
          Ziel-Framerate (CFR der Datei):
        </label>
        <select
          value={targetFps}
          onChange={(e) => onTargetFpsChange(Number(e.target.value))}
          disabled={disabled}
          className="w-full bg-gray-700 text-gray-200 text-sm rounded-md px-2 py-1.5 border border-gray-600 disabled:opacity-50"
        >
          {TARGET_FPS_OPTIONS.map((fps) => (
            <option key={fps} value={fps}>
              {fps} fps
            </option>
          ))}
        </select>
      </div>

      <p className="text-xs text-gray-500 leading-snug">
        Datei extern auf CFR konvertieren, z.B.
        <code className="text-gray-300">
          {" "}
          ffmpeg -i in.mov -vf fps={targetFps} -fps_mode cfr out.mp4
        </code>
      </p>
    </div>
  );
}
