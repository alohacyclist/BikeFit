import { useRef, useEffect, useCallback, useState } from "react";
import { SIDE_KEYPOINTS, type BodySide } from "../utils/AngleCalculator";
import type { Pose } from "../hooks/usePoseDetection";
import { useVideoSource } from "../hooks/useVideoSource";

const SKELETON_CONNECTIONS: [number, number][] = [
  [5, 6],
  [5, 7],
  [7, 9],
  [6, 8],
  [8, 10],
  [5, 11],
  [6, 12],
  [11, 12],
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
];

type AnalysisPhase = "setup" | "recording" | "complete";

interface AnalysisViewProps {
  detectPose: (
    video: HTMLVideoElement,
    opts?: { record?: boolean },
  ) => Promise<Pose | null>;
  isDetectorReady: boolean;
  onCancel: () => void;
  onFrameMeasurement?: (pose: Pose) => void;
  onSideLocked?: (side: BodySide) => void;
  onRecordingFinalize?: (
    durationSeconds: number,
    expectedFrames: number,
  ) => void;
  /** Bei Aufnahmestart aufgerufen — z.B. um das Warmup-Fenster zu starten. */
  onRecordingStart?: () => void;
  /** Meldet aktiven Recording-Zustand (true=läuft) — Parent sperrt Controls. */
  onRecordingActiveChange?: (active: boolean) => void;
  /** Aufnahme verworfen (Abbruch / Seek-Fehler) — Parent setzt Session zurück. */
  onRecordingAbort?: () => void;

  /** Replay-Datei (CFR-konvertiert). */
  videoFile: File | null;
  /** Manuell gewählte Körperseite. */
  forcedSide: BodySide;
  /** Ziel-Framerate für deterministisches Seek-Stepping. */
  targetFps: number;
}

/**
 * Seekt das Video exakt auf t Sekunden und resolved, sobald der Frame
 * dekodiert/präsentiert ist. Grundlage des deterministischen Steppings:
 * jeder Frame wird genau einmal verarbeitet, unabhängig von Inferenz-Tempo.
 */
// Liefert true wenn der Ziel-Frame sauber präsentiert wurde, false bei
// Decode-Fehler oder Timeout. Der Aufrufer MUSS auf false reagieren (Abbruch),
// sonst würde ein falscher/alter Frame still als Frame i verbucht — das bräche
// die Determinismus-Garantie ("jeder Frame genau einmal").
function seekTo(video: HTMLVideoElement, t: number): Promise<boolean> {
  return new Promise((resolve) => {
    // Same-Position-Seek feuert KEIN 'seeked' (Browser-No-op) → sofort lösen,
    // sonst hängt das await (z.B. Preview-Seek auf 0 bei frisch geladener Datei).
    if (Math.abs(video.currentTime - t) < 1e-3 && video.readyState >= 2) {
      resolve(true);
      return;
    }
    let done = false;
    let timer = 0;
    const settle = (ok: boolean) => {
      if (done) return;
      done = true;
      video.removeEventListener("seeked", onSeeked);
      video.removeEventListener("error", onError);
      clearTimeout(timer);
      resolve(ok);
    };
    const onSeeked = () => settle(true);
    const onError = () => settle(false);
    // Sicherheitsnetz: nie permanent hängen. 5s liegt weit über jedem realen
    // Seek (<100ms) — schlägt es zu, ist der Lauf kaputt und wird abgebrochen.
    timer = window.setTimeout(() => settle(false), 5000);
    video.addEventListener("seeked", onSeeked);
    video.addEventListener("error", onError);
    video.currentTime = t;
  });
}

export function AnalysisView({
  detectPose,
  isDetectorReady,
  onCancel,
  onFrameMeasurement,
  onSideLocked,
  onRecordingFinalize,
  onRecordingStart,
  onRecordingActiveChange,
  onRecordingAbort,
  videoFile,
  forcedSide,
  targetFps,
}: AnalysisViewProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const onFrameRef = useRef(onFrameMeasurement);

  const [phase, setPhase] = useState<AnalysisPhase>("setup");
  const [videoProgress, setVideoProgress] = useState(0);
  const [recordingError, setRecordingError] = useState<string | null>(null);
  const [lockedSide, setLockedSide] = useState<BodySide | null>(null);
  const [kpScores, setKpScores] = useState<{
    left: { hip: number; knee: number; ankle: number };
    right: { hip: number; knee: number; ankle: number };
  }>({
    left: { hip: 0, knee: 0, ankle: 0 },
    right: { hip: 0, knee: 0, ankle: 0 },
  });

  useEffect(() => {
    onFrameRef.current = onFrameMeasurement;
  }, [onFrameMeasurement]);

  // Datei-Quelle an das <video>-Element binden.
  const sourceState = useVideoSource(videoRef, {
    file: videoFile,
    targetFps,
  });

  const drawSkeleton = useCallback(
    (
      ctx: CanvasRenderingContext2D,
      keypoints: { x: number; y: number; score?: number }[],
      minConfidence = 0.2,
    ) => {
      ctx.strokeStyle = "#00ff88";
      ctx.lineWidth = 3;
      for (const [s, e] of SKELETON_CONNECTIONS) {
        const a = keypoints[s];
        const b = keypoints[e];
        if (
          a?.score !== undefined &&
          a.score > minConfidence &&
          b?.score !== undefined &&
          b.score > minConfidence
        ) {
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }
      for (const kp of keypoints) {
        if (kp?.score !== undefined && kp.score > minConfidence) {
          ctx.fillStyle = "#ff0066";
          ctx.beginPath();
          ctx.arc(kp.x, kp.y, 6, 0, 2 * Math.PI);
          ctx.fill();
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 2;
          ctx.stroke();
        }
      }
    },
    [],
  );

  const startRecording = useCallback(() => {
    setLockedSide(forcedSide);
    onSideLocked?.(forcedSide);
    setRecordingError(null);
    // Warmup-/Frame-Reset passiert im Recording-Loop (run()), damit jeder
    // (Re-)Lauf konsistent bei frameIndex 1 / Warmup-Fenster 0 startet.
    onRecordingActiveChange?.(true);
    setVideoProgress(0);
    setPhase("recording");
  }, [forcedSide, onSideLocked, onRecordingActiveChange]);

  // Manueller Abbruch: Aufnahme VERWERFEN (nicht finalisieren) — sonst landen
  // Teildaten als gültige Validierungsmetriken im Export. Parent setzt Session
  // via onRecordingAbort zurück.
  const handleStop = useCallback(() => {
    onRecordingActiveChange?.(false);
    onRecordingAbort?.();
    setLockedSide(null);
    setVideoProgress(0);
    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.currentTime = 0;
    }
    setPhase("setup");
  }, [onRecordingActiveChange, onRecordingAbort]);

  // Setup-Preview: ersten Frame mit Skeleton zeigen + Sichtbarkeit je Seite.
  useEffect(() => {
    if (!isDetectorReady || phase !== "setup" || !sourceState.isReady) return;
    let cancelled = false;

    (async () => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas) return;
      await seekTo(video, 0);
      if (cancelled) return;
      const pose = await detectPose(video, { record: false });
      if (cancelled) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      if (pose) {
        drawSkeleton(ctx, pose.keypoints);
        const ixL = SIDE_KEYPOINTS.left;
        const ixR = SIDE_KEYPOINTS.right;
        setKpScores({
          left: {
            hip: pose.keypoints[ixL.hip]?.score ?? 0,
            knee: pose.keypoints[ixL.knee]?.score ?? 0,
            ankle: pose.keypoints[ixL.ankle]?.score ?? 0,
          },
          right: {
            hip: pose.keypoints[ixR.hip]?.score ?? 0,
            knee: pose.keypoints[ixR.knee]?.score ?? 0,
            ankle: pose.keypoints[ixR.ankle]?.score ?? 0,
          },
        });
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [isDetectorReady, phase, sourceState.isReady, detectPose, drawSkeleton]);

  // Recording-Loop: deterministisches Frame-für-Frame-Stepping.
  // Verarbeitet EXAKT floor(duration * targetFps) Frames — identischer
  // Frame-Satz über alle Quantisierungsstufen, unabhängig vom Inferenz-Tempo.
  useEffect(() => {
    if (
      !isDetectorReady ||
      phase !== "recording" ||
      !lockedSide ||
      !sourceState.isReady
    )
      return;
    let cancelled = false;

    const run = async () => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      // Frame-Zähler / Warmup-Fenster im Hook zurücksetzen, damit frameIndex
      // bei 1 startet und keine Frames doppelt gezählt werden
      // (StrictMode / unerwarteter Effekt-Neustart).
      onRecordingStart?.();

      // Verwirft eine fehlgeschlagene/unvollständige Aufnahme statt sie zu
      // finalisieren — Parent setzt die Benchmark-Session zurück.
      const abort = (reason: string) => {
        console.error(reason);
        setRecordingError(reason);
        onRecordingActiveChange?.(false);
        onRecordingAbort?.();
        setLockedSide(null);
        setPhase("setup");
      };

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      const dur = video.duration;
      const total =
        Number.isFinite(dur) && dur > 0 ? Math.floor(dur * targetFps) : 0;

      if (total <= 0) {
        abort("Datei-Dauer nicht lesbar — Aufnahme abgebrochen.");
        return;
      }

      for (let i = 0; i < total && !cancelled; i++) {
        // Frame-Mitte ansteuern, robust gegen Rundung an Frame-Grenzen.
        const ok = await seekTo(video, (i + 0.5) / targetFps);
        if (cancelled) return;
        if (!ok) {
          // Seek-Timeout/Decode-Fehler → kein stiller Falschframe, hart abbrechen.
          abort(`Seek auf Frame ${i} fehlgeschlagen — Aufnahme abgebrochen.`);
          return;
        }
        const pose = await detectPose(video);
        if (cancelled) return;

        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        if (pose) {
          // Reiner Datensammler: Pose an Parent für Export, Skeleton-Render.
          // Winkelberechnung/Statistik passiert im Python-Postprocessing.
          onFrameRef.current?.(pose);
          drawSkeleton(ctx, pose.keypoints);
        }

        setVideoProgress((i + 1) / total);
      }

      if (!cancelled) {
        onRecordingActiveChange?.(false);
        setPhase("complete");
        onRecordingFinalize?.(dur, total);
      }
    };

    run();
    return () => {
      cancelled = true;
    };
  }, [
    isDetectorReady,
    phase,
    lockedSide,
    targetFps,
    sourceState.isReady,
    detectPose,
    drawSkeleton,
    onRecordingFinalize,
    onRecordingActiveChange,
    onRecordingStart,
    onRecordingAbort,
  ]);

  const canStart = isDetectorReady && !!videoFile && sourceState.isReady;

  return (
    <div className="space-y-4">
      <div className="relative w-full max-w-4xl mx-auto">
        <video ref={videoRef} className="hidden" playsInline muted />
        <canvas
          ref={canvasRef}
          className="w-full h-auto rounded-lg shadow-2xl border-2 border-gray-700"
        />

        {phase === "recording" && (
          <div className="absolute top-4 left-4 right-4 flex justify-between items-start pointer-events-none">
            <div className="bg-black/80 backdrop-blur-sm rounded-lg px-4 py-3">
              <div className="text-sm text-gray-400 mb-1">Aufnahme läuft</div>
              <div className="h-2 bg-gray-700 rounded-full overflow-hidden w-48">
                <div
                  className="h-full bg-green-400 transition-all duration-300"
                  style={{ width: `${videoProgress * 100}%` }}
                />
              </div>
              <div className="mt-1 text-xs text-gray-400 tabular-nums">
                Video: {(videoProgress * 100).toFixed(0)}%
              </div>
              <div className="mt-2 text-xs text-gray-300">
                Seite:{" "}
                <span className="font-semibold text-yellow-300">
                  {lockedSide === "right" ? "rechts" : "links"}
                </span>
              </div>
            </div>
          </div>
        )}

        {!isDetectorReady && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/70 rounded-lg">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-green-400 mx-auto mb-4" />
              <p className="text-lg">MoveNet (TFLite) wird geladen...</p>
            </div>
          </div>
        )}

        {sourceState.error && (
          <div className="absolute inset-0 flex items-center justify-center bg-red-900/80 rounded-lg">
            <div className="text-white font-semibold p-6">
              Video-Quelle: {sourceState.error}
            </div>
          </div>
        )}
      </div>

      {recordingError && phase === "setup" && (
        <div className="max-w-4xl mx-auto bg-red-900/50 border border-red-500 rounded-lg px-4 py-3 text-sm text-red-200">
          {recordingError}
        </div>
      )}

      {phase === "setup" && (
        <div className="max-w-4xl mx-auto bg-gray-800/50 rounded-lg p-6 border border-gray-700 space-y-5">
          <div>
            <h3 className="text-lg font-semibold text-green-400 mb-3">
              Sichtbarkeit (erster Frame)
            </h3>
            <div className="grid grid-cols-2 gap-4">
              <SideScorePanel
                title="Links"
                scores={kpScores.left}
                isHighlighted={forcedSide === "left"}
                label={forcedSide === "left" ? "gewählt" : null}
              />
              <SideScorePanel
                title="Rechts"
                scores={kpScores.right}
                isHighlighted={forcedSide === "right"}
                label={forcedSide === "right" ? "gewählt" : null}
              />
            </div>
          </div>

          <div className="border-t border-gray-700 pt-3 text-sm text-gray-400 grid grid-cols-3 gap-x-4">
            <div>
              Quelle:{" "}
              <span className="text-gray-200 font-mono">
                {videoFile ? videoFile.name : "Keine Datei"}
              </span>
            </div>
            <div>
              Seite:{" "}
              <span className="text-yellow-300 font-semibold">
                {forcedSide === "right" ? "rechts" : "links"}
              </span>
            </div>
            <div>
              Frames:{" "}
              <span className="text-gray-200 font-mono">
                {sourceState.totalFrames ?? "–"}
              </span>{" "}
              @ {targetFps} fps
            </div>
            <div>
              Länge:{" "}
              <span className="text-gray-200 font-mono">
                {sourceState.durationSeconds > 0
                  ? `${sourceState.durationSeconds.toFixed(1)} s`
                  : "–"}
              </span>
            </div>
          </div>

          <div className="flex flex-col sm:flex-row gap-3">
            <button
              onClick={startRecording}
              disabled={!canStart}
              className={`flex-1 py-4 px-6 rounded-lg font-bold text-xl transition-all ${
                canStart
                  ? "bg-green-500 hover:bg-green-600 text-white"
                  : "bg-gray-600 text-gray-400 cursor-not-allowed"
              }`}
            >
              Analyse starten
            </button>
            <button
              onClick={onCancel}
              className="py-4 px-6 rounded-lg font-semibold bg-gray-700 hover:bg-gray-600 text-gray-300"
            >
              Abbrechen
            </button>
          </div>
        </div>
      )}

      {phase === "recording" && (
        <div className="max-w-4xl mx-auto flex justify-center">
          <button
            onClick={handleStop}
            className="py-3 px-8 rounded-lg font-semibold bg-red-500 hover:bg-red-600 text-white"
          >
            Analyse abbrechen
          </button>
        </div>
      )}
    </div>
  );
}

interface SideScorePanelProps {
  title: string;
  scores: { hip: number; knee: number; ankle: number };
  isHighlighted: boolean;
  label: string | null;
}

function SideScorePanel({
  title,
  scores,
  isHighlighted,
  label,
}: SideScorePanelProps) {
  return (
    <div
      className={`rounded-lg p-3 border-2 ${
        isHighlighted
          ? "border-yellow-400 bg-yellow-900/20"
          : "border-gray-700 bg-gray-900/40"
      }`}
    >
      <div className="flex items-baseline justify-between mb-2">
        <span className="text-base font-bold text-gray-200">{title}</span>
        {label && (
          <span className="text-xs text-yellow-300 font-semibold">
            ★ {label}
          </span>
        )}
      </div>
      <ScoreRow label="Hüfte" value={scores.hip} />
      <ScoreRow label="Knie" value={scores.knee} />
      <ScoreRow label="Knöchel" value={scores.ankle} />
    </div>
  );
}

function ScoreRow({ label, value }: { label: string; value: number }) {
  const pct = Math.round(value * 100);
  const color = value >= 0.5 ? "#22c55e" : value >= 0.2 ? "#eab308" : "#ef4444";
  return (
    <div className="flex items-center gap-2 py-1">
      <span className="text-sm text-gray-400 w-20">{label}</span>
      <div className="flex-1 h-2 bg-gray-800 rounded-full overflow-hidden">
        <div
          className="h-full transition-all duration-200"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
      <span
        className="text-sm font-mono font-bold tabular-nums w-10 text-right"
        style={{ color }}
      >
        {pct}
      </span>
    </div>
  );
}
