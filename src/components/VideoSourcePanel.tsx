import { useRef, useState } from 'react';
import type { VideoSourceMode } from '../hooks/useVideoSource';
import { VideoRecorder } from '../services/VideoRecorder';

interface VideoSourcePanelProps {
  mode: VideoSourceMode;
  onModeChange: (mode: VideoSourceMode) => void;
  file: File | null;
  onFileChange: (file: File | null) => void;
  /** Stream für laufende Aufnahme — muss vom Aufrufer aus dem Webcam-Modus stammen. */
  webcamStream?: MediaStream | null;
  participantId: string;
}

/**
 * UI-Panel für die Auswahl der Video-Quelle (Webcam vs Datei-Replay).
 * Erlaubt zusätzlich, im Webcam-Modus eine Aufnahme zu starten und als
 * .webm herunterzuladen — diese kann anschließend (CFR-konvertiert) als
 * Replay-Datei geladen werden.
 */
export function VideoSourcePanel({
  mode,
  onModeChange,
  file,
  onFileChange,
  webcamStream,
  participantId,
}: VideoSourcePanelProps) {
  const recorderRef = useRef<VideoRecorder>(new VideoRecorder());
  const [isRecording, setIsRecording] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0] ?? null;
    onFileChange(f);
  };

  const startRecording = () => {
    setError(null);
    if (!webcamStream) {
      setError('Webcam-Stream nicht verfügbar');
      return;
    }
    try {
      recorderRef.current.start(webcamStream);
      setIsRecording(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Aufnahme-Start fehlgeschlagen');
    }
  };

  const stopRecording = async () => {
    try {
      const blob = await recorderRef.current.stop();
      recorderRef.current.downloadBlob(blob, participantId);
      setIsRecording(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Aufnahme-Stop fehlgeschlagen');
      setIsRecording(false);
    }
  };

  return (
    <div className="bg-gray-800/60 rounded-lg p-4 border border-gray-700 space-y-3">
      <h3 className="text-sm font-semibold text-gray-300">Video-Quelle</h3>

      <div className="grid grid-cols-2 gap-2">
        <button
          onClick={() => onModeChange('webcam')}
          className={`py-2 px-3 rounded-md text-sm font-semibold transition-colors ${
            mode === 'webcam'
              ? 'bg-blue-600 text-white border border-blue-400'
              : 'bg-gray-700 text-gray-300 border border-gray-600 hover:bg-gray-600'
          }`}
        >
          Webcam
        </button>
        <button
          onClick={() => onModeChange('file')}
          className={`py-2 px-3 rounded-md text-sm font-semibold transition-colors ${
            mode === 'file'
              ? 'bg-blue-600 text-white border border-blue-400'
              : 'bg-gray-700 text-gray-300 border border-gray-600 hover:bg-gray-600'
          }`}
        >
          Replay-Datei
        </button>
      </div>

      {mode === 'webcam' && (
        <div className="space-y-2">
          <div className="text-xs text-gray-400">
            Live-Modus: Setup + Aufnahme der Probandenbewegung als Referenz-Video.
          </div>
          {!isRecording ? (
            <button
              onClick={startRecording}
              disabled={!webcamStream}
              className={`w-full py-2 px-3 rounded-md text-sm font-semibold ${
                webcamStream
                  ? 'bg-red-600 hover:bg-red-700 text-white'
                  : 'bg-gray-700 text-gray-500 cursor-not-allowed'
              }`}
            >
              ● Aufnahme starten
            </button>
          ) : (
            <button
              onClick={stopRecording}
              className="w-full py-2 px-3 rounded-md text-sm font-semibold bg-gray-200 hover:bg-white text-gray-900"
            >
              ■ Stop &amp; Download
            </button>
          )}
          <p className="text-xs text-gray-500 leading-snug">
            Hinweis: WebM-Aufnahme ist VFR. Vor Replay-Nutzung mit ffmpeg auf
            CFR konvertieren (z.B. <code className="text-gray-300">-r 30 -vsync cfr -g 1</code>).
          </p>
        </div>
      )}

      {mode === 'file' && (
        <div className="space-y-2">
          <label className="block text-xs text-gray-400">
            Replay-Video laden (.mp4 / .webm, CFR empfohlen):
          </label>
          <input
            type="file"
            accept="video/mp4,video/webm,video/*"
            onChange={handleFile}
            className="block w-full text-xs text-gray-300 file:mr-2 file:py-1 file:px-3 file:rounded-md file:border-0 file:bg-blue-600 file:text-white hover:file:bg-blue-700"
          />
          {file && (
            <div className="text-xs text-gray-400">
              Geladen: <span className="text-gray-200">{file.name}</span> (
              {(file.size / 1024 / 1024).toFixed(1)} MB)
            </div>
          )}
        </div>
      )}

      {error && <div className="text-xs text-red-400">{error}</div>}
    </div>
  );
}
