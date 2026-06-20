import { useRef, useEffect, useCallback, useState } from 'react';
import {
  calculateKneeAngle,
  getAngleColor,
  isKneeTripleValid,
  SIDE_KEYPOINTS,
  type BodySide,
  BiomechanicalAngles,
  type Keypoint,
} from '../utils/AngleCalculator';
import {
  BiomechanicalAnalyzer,
  AnalysisResults,
  type CyclePhase,
} from '../services/BiomechanicalAnalyzer';
import type { Pose } from '../hooks/usePoseDetection';
import {
  useVideoSource,
  type VideoSourceMode,
} from '../hooks/useVideoSource';

const SKELETON_CONNECTIONS: [number, number][] = [
  [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
  [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
];

type AnalysisPhase = 'setup' | 'countdown' | 'recording' | 'complete';

const COUNTDOWN_SECONDS = 15;
const SETUP_SIDE_WINDOW = 30;
const RECORDING_AUTOSTOP_MS = 60000;

interface AnalysisViewProps {
  detectPose: (video: HTMLVideoElement) => Promise<Pose | null>;
  isDetectorReady: boolean;
  onComplete: (results: AnalysisResults) => void;
  onCancel: () => void;
  targetCycles?: number;
  onFrameMeasurement?: (pose: Pose) => void;
  onSideLocked?: (side: BodySide) => void;
  onRecordingFinalize?: (interpolatedFrames: number, cyclesDetected: number) => void;

  /** Video-Quelle (Webcam oder Datei-Replay). */
  videoMode: VideoSourceMode;
  videoFile: File | null;
  /** Wenn gesetzt: überschreibt die runtime-Auto-Detection für Side-Lock. */
  forcedSide?: BodySide | null;
  /** Im File-Modus: Countdown überspringen (Auto-Start). */
  skipCountdownInFileMode?: boolean;
}

function beep(freq: number, durationMs: number, gain = 0.15): void {
  try {
    const Ctx =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext: typeof AudioContext })
        .webkitAudioContext;
    const ctx = new Ctx();
    const osc = ctx.createOscillator();
    const g = ctx.createGain();
    osc.frequency.value = freq;
    g.gain.value = gain;
    osc.connect(g).connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + durationMs / 1000);
    osc.onended = () => ctx.close();
  } catch {
    // no audio
  }
}

export function AnalysisView({
  detectPose,
  isDetectorReady,
  onComplete,
  onCancel,
  targetCycles = 5,
  onFrameMeasurement,
  onSideLocked,
  onRecordingFinalize,
  videoMode,
  videoFile,
  forcedSide,
  skipCountdownInFileMode = true,
}: AnalysisViewProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>(0);
  const analyzerRef = useRef<BiomechanicalAnalyzer | null>(null);
  const onFrameRef = useRef(onFrameMeasurement);
  const recordingStartRef = useRef<number>(0);

  const sideHistoryRef = useRef<{ left: number[]; right: number[] }>({
    left: [],
    right: [],
  });

  const [phase, setPhase] = useState<AnalysisPhase>('setup');
  const [countdown, setCountdown] = useState<number>(COUNTDOWN_SECONDS);
  const [cycleCount, setCycleCount] = useState(0);
  const [cyclePhase, setCyclePhase] = useState<CyclePhase>('searching');
  const [currentAngles, setCurrentAngles] = useState<BiomechanicalAngles>({
    knee: null,
  });
  const [lockedSide, setLockedSide] = useState<BodySide | null>(null);
  const [recommendedSide, setRecommendedSide] =
    useState<BodySide>('right');
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

  // Video-Quelle aktiv halten (Webcam-Stream oder File-Backed)
  const sourceState = useVideoSource(videoRef, {
    mode: videoMode,
    file: videoFile,
  });

  useEffect(() => {
    analyzerRef.current = new BiomechanicalAnalyzer({ targetCycles });
    return () => {
      analyzerRef.current = null;
    };
  }, [targetCycles]);

  // Countdown — nur im Webcam-Modus, im File-Modus optional überspringbar
  useEffect(() => {
    if (phase !== 'countdown') return;
    setCountdown(COUNTDOWN_SECONDS);
    let remaining = COUNTDOWN_SECONDS;
    beep(660, 120);
    const id = setInterval(() => {
      remaining -= 1;
      setCountdown(remaining);
      if (remaining > 0 && remaining <= 3) beep(660, 120);
      if (remaining <= 0) {
        clearInterval(id);
        beep(990, 250);
        startRecording();
      }
    }, 1000);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase]);

  const lockSide = useCallback((): BodySide => {
    if (forcedSide) return forcedSide;
    const hist = sideHistoryRef.current;
    const meanL =
      hist.left.length === 0
        ? 0
        : hist.left.reduce((a, b) => a + b, 0) / hist.left.length;
    const meanR =
      hist.right.length === 0
        ? 0
        : hist.right.reduce((a, b) => a + b, 0) / hist.right.length;
    return meanR >= meanL ? 'right' : 'left';
  }, [forcedSide]);

  const startRecording = useCallback(() => {
    const side = lockSide();
    setLockedSide(side);
    onSideLocked?.(side);
    analyzerRef.current?.reset();
    setCycleCount(0);
    recordingStartRef.current = performance.now();
    // Im File-Modus: Video von vorne starten und abspielen
    if (videoMode === 'file' && videoRef.current) {
      videoRef.current.currentTime = 0;
      videoRef.current
        .play()
        .catch((e) => console.error('Video-Play fehlgeschlagen:', e));
    }
    setPhase('recording');
  }, [lockSide, onSideLocked, videoMode]);

  const drawSkeleton = useCallback(
    (
      ctx: CanvasRenderingContext2D,
      keypoints: { x: number; y: number; score?: number }[],
      minConfidence = 0.2
    ) => {
      ctx.strokeStyle = '#00ff88';
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
          ctx.fillStyle = '#ff0066';
          ctx.beginPath();
          ctx.arc(kp.x, kp.y, 6, 0, 2 * Math.PI);
          ctx.fill();
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 2;
          ctx.stroke();
        }
      }
    },
    []
  );

  const drawAngleOverlay = useCallback(
    (
      ctx: CanvasRenderingContext2D,
      pos: { x: number; y: number },
      angle: number,
      label: string
    ) => {
      const color = getAngleColor(angle);
      const text = `${label}: ${angle.toFixed(0)}°`;
      ctx.font = 'bold 14px monospace';
      const m = ctx.measureText(text);
      const p = 4;
      ctx.fillStyle = 'rgba(0,0,0,0.8)';
      ctx.fillRect(pos.x - p, pos.y - 16 - p, m.width + p * 2, 20 + p);
      ctx.fillStyle = color;
      ctx.fillText(text, pos.x, pos.y - 4);
    },
    []
  );

  // Recording-Loop
  useEffect(() => {
    if (!isDetectorReady || phase !== 'recording' || !lockedSide) return;

    const renderLoop = async () => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const analyzer = analyzerRef.current;
      if (!video || !canvas || !analyzer || video.readyState < 2) {
        animationRef.current = requestAnimationFrame(renderLoop);
        return;
      }
      if (
        canvas.width !== video.videoWidth ||
        canvas.height !== video.videoHeight
      ) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        animationRef.current = requestAnimationFrame(renderLoop);
        return;
      }

      const pose = await detectPose(video);

      ctx.save();
      if (videoMode === 'webcam') {
        ctx.scale(-1, 1);
        ctx.drawImage(
          video,
          -canvas.width,
          0,
          canvas.width,
          canvas.height
        );
      } else {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      }
      ctx.restore();

      if (pose) {
        if (onFrameRef.current) onFrameRef.current(pose);

        // Im Webcam-Modus spiegeln, im File-Modus 1:1
        const drawKeypoints =
          videoMode === 'webcam'
            ? pose.keypoints.map((kp) => ({
                ...kp,
                x: canvas.width - kp.x,
              }))
            : pose.keypoints;

        drawSkeleton(ctx, drawKeypoints);

        const ix = SIDE_KEYPOINTS[lockedSide];
        // Knie-Winkel auf den ORIGINAL-Keypoints rechnen, nicht gespiegelt
        const hip = pose.keypoints[ix.hip] as Keypoint;
        const knee = pose.keypoints[ix.knee] as Keypoint;
        const ankle = pose.keypoints[ix.ankle] as Keypoint;

        const angles: BiomechanicalAngles = {
          knee: calculateKneeAngle(hip, knee, ankle),
        };

        const relevant = [hip, knee, ankle];
        const avgConfidence =
          relevant.reduce((s, k) => s + (k.score || 0), 0) / relevant.length;

        analyzer.addFrame(angles, avgConfidence);
        setCurrentAngles(angles);
        setCycleCount(analyzer.getCycleCount());
        setCyclePhase(analyzer.getCurrentPhase());

        if (angles.knee !== null) {
          const drawKnee = drawKeypoints[ix.knee];
          drawAngleOverlay(
            ctx,
            { x: drawKnee.x + 15, y: drawKnee.y },
            angles.knee,
            'Knie'
          );
        }

        const elapsed = performance.now() - recordingStartRef.current;
        const videoEnded = videoMode === 'file' && video.ended;
        if (
          analyzer.isComplete() ||
          elapsed > RECORDING_AUTOSTOP_MS ||
          videoEnded
        ) {
          const results = analyzer.getResults();
          setPhase('complete');
          onRecordingFinalize?.(
            analyzer.getInterpolatedFramesCount(),
            analyzer.getCycleCount()
          );
          if (results) onComplete(results);
          return;
        }
      }

      animationRef.current = requestAnimationFrame(renderLoop);
    };

    animationRef.current = requestAnimationFrame(renderLoop);
    return () => cancelAnimationFrame(animationRef.current);
  }, [
    isDetectorReady,
    detectPose,
    phase,
    lockedSide,
    videoMode,
    drawSkeleton,
    drawAngleOverlay,
    onComplete,
    onRecordingFinalize,
  ]);

  // Setup-Loop (Live-Preview + Side-Quality)
  useEffect(() => {
    if (
      !isDetectorReady ||
      (phase !== 'setup' && phase !== 'countdown')
    )
      return;

    const renderLoop = async () => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas || video.readyState < 2) {
        animationRef.current = requestAnimationFrame(renderLoop);
        return;
      }
      if (
        canvas.width !== video.videoWidth ||
        canvas.height !== video.videoHeight
      ) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        animationRef.current = requestAnimationFrame(renderLoop);
        return;
      }

      const pose = await detectPose(video);

      ctx.save();
      if (videoMode === 'webcam') {
        ctx.scale(-1, 1);
        ctx.drawImage(
          video,
          -canvas.width,
          0,
          canvas.width,
          canvas.height
        );
      } else {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      }
      ctx.restore();

      if (pose) {
        const drawKeypoints =
          videoMode === 'webcam'
            ? pose.keypoints.map((kp) => ({
                ...kp,
                x: canvas.width - kp.x,
              }))
            : pose.keypoints;
        drawSkeleton(ctx, drawKeypoints);

        const vL = isKneeTripleValid(pose.keypoints, 'left');
        const vR = isKneeTripleValid(pose.keypoints, 'right');
        const hist = sideHistoryRef.current;
        hist.left.push(vL);
        hist.right.push(vR);
        if (hist.left.length > SETUP_SIDE_WINDOW) hist.left.shift();
        if (hist.right.length > SETUP_SIDE_WINDOW) hist.right.shift();
        const meanL =
          hist.left.reduce((a, b) => a + b, 0) / hist.left.length;
        const meanR =
          hist.right.reduce((a, b) => a + b, 0) / hist.right.length;
        setRecommendedSide(meanR >= meanL ? 'right' : 'left');

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

      animationRef.current = requestAnimationFrame(renderLoop);
    };

    animationRef.current = requestAnimationFrame(renderLoop);
    return () => cancelAnimationFrame(animationRef.current);
  }, [isDetectorReady, detectPose, phase, drawSkeleton, videoMode]);

  const handleStart = () => {
    if (videoMode === 'file' && skipCountdownInFileMode) {
      // Datei-Replay: kein Countdown nötig
      startRecording();
    } else {
      setPhase('countdown');
    }
  };

  const handleStop = () => {
    const analyzer = analyzerRef.current;
    if (analyzer) {
      onRecordingFinalize?.(
        analyzer.getInterpolatedFramesCount(),
        analyzer.getCycleCount()
      );
      analyzer.reset();
    }
    setLockedSide(null);
    setCycleCount(0);
    if (videoMode === 'file' && videoRef.current) {
      videoRef.current.pause();
      videoRef.current.currentTime = 0;
    }
    setPhase('setup');
  };

  const sourceLabel =
    videoMode === 'webcam'
      ? 'Webcam'
      : videoFile
      ? videoFile.name
      : 'Keine Datei';

  return (
    <div className="space-y-4">
      <div className="relative w-full max-w-4xl mx-auto">
        <video ref={videoRef} className="hidden" playsInline muted />
        <canvas
          ref={canvasRef}
          className="w-full h-auto rounded-lg shadow-2xl border-2 border-gray-700"
        />

        {phase === 'countdown' && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/70 rounded-lg">
            <div className="text-9xl font-black text-yellow-400 tabular-nums drop-shadow-[0_0_20px_rgba(0,0,0,0.8)]">
              {countdown}
            </div>
            <div className="mt-4 text-2xl text-white font-semibold">
              Auf Position gehen
            </div>
            <button
              onClick={() => setPhase('setup')}
              className="mt-6 py-2 px-6 rounded-lg bg-gray-700/80 hover:bg-gray-600 text-gray-200"
            >
              Abbrechen
            </button>
          </div>
        )}

        {phase === 'recording' && (
          <div className="absolute top-4 left-4 right-4 flex justify-between items-start pointer-events-none">
            <div className="bg-black/80 backdrop-blur-sm rounded-lg px-4 py-3">
              <div className="text-sm text-gray-400 mb-1">Aufnahme</div>
              <div className="flex items-center gap-3">
                <div className="text-3xl font-bold text-green-400 tabular-nums">
                  {cycleCount} / {targetCycles}
                </div>
                <div className="text-sm text-gray-400">Zyklen</div>
              </div>
              <div className="mt-2 h-2 bg-gray-700 rounded-full overflow-hidden w-48">
                <div
                  className="h-full bg-green-400 transition-all duration-300"
                  style={{
                    width: `${(cycleCount / targetCycles) * 100}%`,
                  }}
                />
              </div>
              <div className="mt-2 text-xs text-gray-300">
                Seite:{' '}
                <span className="font-semibold text-yellow-300">
                  {lockedSide === 'right' ? 'rechts' : 'links'}
                </span>
              </div>
              <div className="text-xs text-gray-300">
                Status:{' '}
                <span
                  className={
                    cyclePhase === 'searching'
                      ? 'text-yellow-300'
                      : 'text-green-300'
                  }
                >
                  {cyclePhase === 'searching'
                    ? 'Warte auf Bewegung'
                    : cyclePhase === 'extension'
                    ? 'Streckung'
                    : 'Beugung'}
                </span>
              </div>
              <div className="text-xs text-gray-300">
                Knie:{' '}
                <span className="font-mono text-green-300">
                  {currentAngles.knee !== null
                    ? `${currentAngles.knee.toFixed(0)}°`
                    : '--'}
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

      {phase === 'setup' && (
        <div className="max-w-4xl mx-auto bg-gray-800/50 rounded-lg p-6 border border-gray-700 space-y-5">
          <div>
            <h3 className="text-lg font-semibold text-green-400 mb-3">
              Sichtbarkeit (Live)
            </h3>
            <div className="grid grid-cols-2 gap-4">
              <SideScorePanel
                title="Links"
                scores={kpScores.left}
                isHighlighted={
                  forcedSide
                    ? forcedSide === 'left'
                    : recommendedSide === 'left'
                }
                label={
                  forcedSide === 'left'
                    ? 'fixiert'
                    : recommendedSide === 'left' && !forcedSide
                    ? 'auto'
                    : null
                }
              />
              <SideScorePanel
                title="Rechts"
                scores={kpScores.right}
                isHighlighted={
                  forcedSide
                    ? forcedSide === 'right'
                    : recommendedSide === 'right'
                }
                label={
                  forcedSide === 'right'
                    ? 'fixiert'
                    : recommendedSide === 'right' && !forcedSide
                    ? 'auto'
                    : null
                }
              />
            </div>
          </div>

          <div className="border-t border-gray-700 pt-3 text-sm text-gray-400 grid grid-cols-2 gap-x-4">
            <div>
              Quelle:{' '}
              <span className="text-gray-200 font-mono">{sourceLabel}</span>
            </div>
            <div>
              Seite:{' '}
              <span className="text-yellow-300 font-semibold">
                {forcedSide
                  ? forcedSide === 'right'
                    ? 'rechts (fix)'
                    : 'links (fix)'
                  : recommendedSide === 'right'
                  ? 'rechts (auto)'
                  : 'links (auto)'}
              </span>
            </div>
          </div>

          <div className="flex flex-col sm:flex-row gap-3">
            <button
              onClick={handleStart}
              disabled={
                !isDetectorReady ||
                (videoMode === 'file' && !videoFile) ||
                !sourceState.isReady
              }
              className={`flex-1 py-4 px-6 rounded-lg font-bold text-xl transition-all ${
                isDetectorReady &&
                sourceState.isReady &&
                (videoMode === 'webcam' || videoFile)
                  ? 'bg-green-500 hover:bg-green-600 text-white'
                  : 'bg-gray-600 text-gray-400 cursor-not-allowed'
              }`}
            >
              {videoMode === 'file' && skipCountdownInFileMode
                ? 'Replay starten'
                : `Aufnahme in ${COUNTDOWN_SECONDS}s`}
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

      {phase === 'recording' && (
        <div className="max-w-4xl mx-auto flex justify-center">
          <button
            onClick={handleStop}
            className="py-3 px-8 rounded-lg font-semibold bg-red-500 hover:bg-red-600 text-white"
          >
            Aufnahme abbrechen
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
          ? 'border-yellow-400 bg-yellow-900/20'
          : 'border-gray-700 bg-gray-900/40'
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
  const color =
    value >= 0.5 ? '#22c55e' : value >= 0.2 ? '#eab308' : '#ef4444';
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
