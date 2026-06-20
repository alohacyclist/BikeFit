import { RefObject, useEffect, useState } from 'react';

/**
 * Video-Quelle für die Pose-Detection.
 * 'webcam'  — Live-Stream via getUserMedia (Default-Workflow, Setup)
 * 'file'    — Replay einer aufgezeichneten Datei (Messmodus, frame-deterministisch)
 */
export type VideoSourceMode = 'webcam' | 'file';

export interface VideoSourceConfig {
  mode: VideoSourceMode;
  file?: File | null;
  /** Constraints nur für Webcam-Modus relevant. */
  constraints?: MediaStreamConstraints;
}

export interface UseVideoSourceResult {
  isReady: boolean;
  error: string | null;
  /** Frame-Anzahl, falls aus Datei und Metadata bereits gelesen wurde (sonst null). */
  totalFrames: number | null;
  /** Echte Dauer in Sekunden (Datei) bzw. 0 (Webcam). */
  durationSeconds: number;
  /** Naturalrate (Datei) bzw. 30 (Webcam-Fallback). */
  nominalFps: number;
}

/**
 * Hängt eine Video-Quelle an ein externes <video>-Element.
 * Cleanup bei Wechsel oder Unmount ist enthalten.
 *
 * Datei-Modus geht davon aus, dass die Aufnahme als CFR (constant frame rate)
 * konvertiert wurde — sonst ist frame-genaues Stepping nicht zuverlässig.
 */
export function useVideoSource(
  videoRef: RefObject<HTMLVideoElement>,
  config: VideoSourceConfig
): UseVideoSourceResult {
  const [isReady, setIsReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [totalFrames, setTotalFrames] = useState<number | null>(null);
  const [durationSeconds, setDurationSeconds] = useState<number>(0);
  const [nominalFps, setNominalFps] = useState<number>(30);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    let cancelled = false;
    let objectUrl: string | null = null;
    let stream: MediaStream | null = null;

    setIsReady(false);
    setError(null);
    setTotalFrames(null);
    setDurationSeconds(0);

    async function setupWebcam() {
      try {
        const constraints =
          config.constraints ?? {
            video: {
              width: { ideal: 1280 },
              height: { ideal: 720 },
              frameRate: { ideal: 30, max: 60 },
            },
            audio: false,
          };
        stream = await navigator.mediaDevices.getUserMedia(constraints);
        if (cancelled || !video) return;
        video.srcObject = stream;
        video.src = '';
        await new Promise<void>((resolve) => {
          video.onloadedmetadata = () => resolve();
        });
        await video.play();
        // Webcam: Nominal-FPS aus Track-Settings extrahieren
        const track = stream.getVideoTracks()[0];
        const settings = track?.getSettings();
        const fps = settings?.frameRate ?? 30;
        setNominalFps(fps);
        setDurationSeconds(0);
        setTotalFrames(null);
        setIsReady(true);
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : 'Webcam nicht verfügbar');
        }
      }
    }

    async function setupFile() {
      try {
        if (!config.file) {
          setError('Keine Datei ausgewählt');
          return;
        }
        objectUrl = URL.createObjectURL(config.file);
        if (!video) return;
        video.srcObject = null;
        video.src = objectUrl;
        // Wichtig: pausiert lassen — Recording-Loop pumpt selbst
        await new Promise<void>((resolve, reject) => {
          video.onloadedmetadata = () => resolve();
          video.onerror = () => reject(new Error('Video-Datei konnte nicht geladen werden'));
        });
        if (cancelled) return;
        video.pause();
        video.currentTime = 0;
        // FPS-Erkennung: HTMLVideoElement bietet keine native FPS-API.
        // Annahme: CFR mit 30 fps falls nichts anderes bekannt.
        // Realistischer Wert wird im Konsumenten via requestVideoFrameCallback
        // anhand presented-Frames bestimmt.
        const dur = video.duration;
        setDurationSeconds(Number.isFinite(dur) ? dur : 0);
        // Konvention: 30 fps annehmen; Replay-Loop nutzt requestVideoFrameCallback
        const fps = 30;
        setNominalFps(fps);
        setTotalFrames(
          Number.isFinite(dur) ? Math.round(dur * fps) : null
        );
        setIsReady(true);
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : 'Datei-Setup fehlgeschlagen');
        }
      }
    }

    if (config.mode === 'webcam') {
      setupWebcam();
    } else {
      setupFile();
    }

    return () => {
      cancelled = true;
      if (stream) {
        stream.getTracks().forEach((t) => t.stop());
      }
      if (objectUrl) {
        URL.revokeObjectURL(objectUrl);
      }
      if (video) {
        video.srcObject = null;
        video.removeAttribute('src');
        video.load();
      }
    };
  }, [config.mode, config.file, config.constraints, videoRef]);

  return { isReady, error, totalFrames, durationSeconds, nominalFps };
}
