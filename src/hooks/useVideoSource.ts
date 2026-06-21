import { RefObject, useEffect, useState } from "react";

/**
 * Video-Quelle für die Pose-Detection — ausschließlich Datei-Replay.
 *
 * Das Benchmark arbeitet bewusst nur mit vorab aufgenommenen, auf CFR
 * (constant frame rate) konvertierten Dateien: nur so ist der Frame-Satz
 * über alle Quantisierungsstufen identisch und das Ergebnis reproduzierbar.
 * Ein Live-Webcam-Pfad würde Nichtdeterminismus (Refresh-Rate, fps-Jitter)
 * einführen und wurde deshalb entfernt.
 */
export interface VideoSourceConfig {
  file: File | null;
  /** Ziel-Framerate (CFR-Annahme der konvertierten Datei). */
  targetFps: number;
}

export interface UseVideoSourceResult {
  isReady: boolean;
  error: string | null;
  /** Erwartete Frame-Anzahl = floor(duration * targetFps), null vor Load.
   *  Muss identisch zur Loop-Berechnung in AnalysisView sein. */
  totalFrames: number | null;
  /** Dauer der Datei in Sekunden (0 vor Load). */
  durationSeconds: number;
  /** Verwendete Ziel-Framerate (== config.targetFps). */
  nominalFps: number;
}

/**
 * Hängt eine Replay-Datei an ein externes <video>-Element.
 * Lässt das Video pausiert — der Recording-Loop steppt selbst via currentTime.
 * Cleanup bei Wechsel oder Unmount ist enthalten.
 */
export function useVideoSource(
  videoRef: RefObject<HTMLVideoElement>,
  config: VideoSourceConfig,
): UseVideoSourceResult {
  const [isReady, setIsReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [totalFrames, setTotalFrames] = useState<number | null>(null);
  const [durationSeconds, setDurationSeconds] = useState<number>(0);

  const { file, targetFps } = config;

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    let cancelled = false;
    let objectUrl: string | null = null;

    setIsReady(false);
    setError(null);
    setTotalFrames(null);
    setDurationSeconds(0);

    async function setupFile() {
      try {
        if (!file) {
          setError("Keine Datei ausgewählt");
          return;
        }
        objectUrl = URL.createObjectURL(file);
        if (!video) return;
        video.srcObject = null;
        video.src = objectUrl;
        // Pausiert lassen — Recording-Loop steppt currentTime selbst.
        await new Promise<void>((resolve, reject) => {
          video.onloadedmetadata = () => resolve();
          video.onerror = () =>
            reject(new Error("Video-Datei konnte nicht geladen werden"));
        });
        if (cancelled) return;
        video.pause();
        video.currentTime = 0;

        const dur = video.duration;
        setDurationSeconds(Number.isFinite(dur) ? dur : 0);
        setTotalFrames(
          Number.isFinite(dur) ? Math.floor(dur * targetFps) : null,
        );
        setIsReady(true);
      } catch (e) {
        if (!cancelled) {
          setError(
            e instanceof Error ? e.message : "Datei-Setup fehlgeschlagen",
          );
        }
      }
    }

    setupFile();

    return () => {
      cancelled = true;
      if (objectUrl) {
        URL.revokeObjectURL(objectUrl);
      }
      if (video) {
        video.srcObject = null;
        video.removeAttribute("src");
        video.load();
      }
    };
  }, [file, targetFps, videoRef]);

  return {
    isReady,
    error,
    totalFrames,
    durationSeconds,
    nominalFps: targetFps,
  };
}
