import { useRef, useEffect, useCallback, useState } from 'react';
import * as poseDetection from '@tensorflow-models/pose-detection';
import {
  calculateKneeAngle,
  calculateHipAngle,
  calculateAnkleAngle,
  calculateElbowAngle,
  calculateBackAngle,
  getAngleColor,
  BiomechanicalAngles,
  type Keypoint,
} from '../utils/AngleCalculator';
import {
  BiomechanicalAnalyzer,
  AnalysisResults,
} from '../services/BiomechanicalAnalyzer';
import type { Pose } from '../hooks/usePoseDetection';

// MoveNet Keypoint-Indizes
const KEYPOINT_INDICES = {
  nose: 0,
  leftEye: 1,
  rightEye: 2,
  leftEar: 3,
  rightEar: 4,
  leftShoulder: 5,
  rightShoulder: 6,
  leftElbow: 7,
  rightElbow: 8,
  leftWrist: 9,
  rightWrist: 10,
  leftHip: 11,
  rightHip: 12,
  leftKnee: 13,
  rightKnee: 14,
  leftAnkle: 15,
  rightAnkle: 16,
};

const SKELETON_CONNECTIONS: [number, number][] = [
  [KEYPOINT_INDICES.leftShoulder, KEYPOINT_INDICES.rightShoulder],
  [KEYPOINT_INDICES.leftShoulder, KEYPOINT_INDICES.leftElbow],
  [KEYPOINT_INDICES.leftElbow, KEYPOINT_INDICES.leftWrist],
  [KEYPOINT_INDICES.rightShoulder, KEYPOINT_INDICES.rightElbow],
  [KEYPOINT_INDICES.rightElbow, KEYPOINT_INDICES.rightWrist],
  [KEYPOINT_INDICES.leftShoulder, KEYPOINT_INDICES.leftHip],
  [KEYPOINT_INDICES.rightShoulder, KEYPOINT_INDICES.rightHip],
  [KEYPOINT_INDICES.leftHip, KEYPOINT_INDICES.rightHip],
  [KEYPOINT_INDICES.leftHip, KEYPOINT_INDICES.leftKnee],
  [KEYPOINT_INDICES.leftKnee, KEYPOINT_INDICES.leftAnkle],
  [KEYPOINT_INDICES.rightHip, KEYPOINT_INDICES.rightKnee],
  [KEYPOINT_INDICES.rightKnee, KEYPOINT_INDICES.rightAnkle],
];

type AnalysisPhase = 'setup' | 'recording' | 'complete';

interface AnalysisViewProps {
  detectPose: (video: HTMLVideoElement) => Promise<Pose | null>;
  isDetectorReady: boolean;
  onComplete: (results: AnalysisResults) => void;
  onCancel: () => void;
  targetCycles?: number;
  // Optionaler Frame-Hook für Benchmark-Recording (nur während 'recording')
  onFrameMeasurement?: (pose: Pose) => void;
}

export function AnalysisView({
  detectPose,
  isDetectorReady,
  onComplete,
  onCancel,
  targetCycles = 5,
  onFrameMeasurement,
}: AnalysisViewProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>(0);
  const analyzerRef = useRef<BiomechanicalAnalyzer | null>(null);
  const onFrameRef = useRef(onFrameMeasurement);

  const [phase, setPhase] = useState<AnalysisPhase>('setup');
  const [cycleCount, setCycleCount] = useState(0);
  const [currentAngles, setCurrentAngles] = useState<BiomechanicalAngles>({
    knee: null,
    hip: null,
    ankle: null,
    elbow: null,
    back: null,
  });
  const [poseConfidence, setPoseConfidence] = useState(0);

  useEffect(() => {
    onFrameRef.current = onFrameMeasurement;
  }, [onFrameMeasurement]);

  // Webcam initialisieren
  useEffect(() => {
    async function setupCamera() {
      if (!videoRef.current) return;
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            frameRate: { ideal: 30, max: 60 },
          },
          audio: false,
        });

        videoRef.current.srcObject = stream;
        await new Promise<void>((resolve) => {
          if (videoRef.current) {
            videoRef.current.onloadedmetadata = () => resolve();
          }
        });
        await videoRef.current.play();
      } catch (err) {
        console.error('Webcam Zugriff fehlgeschlagen:', err);
      }
    }

    setupCamera();

    return () => {
      if (videoRef.current?.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  useEffect(() => {
    analyzerRef.current = new BiomechanicalAnalyzer({ targetCycles });
    return () => {
      analyzerRef.current = null;
    };
  }, [targetCycles]);

  const drawSkeleton = useCallback(
    (
      ctx: CanvasRenderingContext2D,
      keypoints: poseDetection.Keypoint[],
      minConfidence: number = 0.3
    ) => {
      ctx.strokeStyle = '#00ff88';
      ctx.lineWidth = 3;

      for (const [startIdx, endIdx] of SKELETON_CONNECTIONS) {
        const start = keypoints[startIdx];
        const end = keypoints[endIdx];

        if (
          start.score !== undefined &&
          start.score > minConfidence &&
          end.score !== undefined &&
          end.score > minConfidence
        ) {
          ctx.beginPath();
          ctx.moveTo(start.x, start.y);
          ctx.lineTo(end.x, end.y);
          ctx.stroke();
        }
      }

      for (const keypoint of keypoints) {
        if (keypoint.score !== undefined && keypoint.score > minConfidence) {
          ctx.fillStyle = '#ff0066';
          ctx.beginPath();
          ctx.arc(keypoint.x, keypoint.y, 6, 0, 2 * Math.PI);
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
      position: { x: number; y: number },
      angle: number,
      label: string
    ) => {
      const color = getAngleColor(angle);
      const text = `${label}: ${angle.toFixed(0)}°`;

      ctx.font = 'bold 14px monospace';
      const textMetrics = ctx.measureText(text);
      const padding = 4;

      ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
      ctx.fillRect(
        position.x - padding,
        position.y - 16 - padding,
        textMetrics.width + padding * 2,
        20 + padding
      );

      ctx.fillStyle = color;
      ctx.fillText(text, position.x, position.y - 4);
    },
    []
  );

  // Recording-Loop
  useEffect(() => {
    if (!isDetectorReady || phase !== 'recording') return;

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
      ctx.scale(-1, 1);
      ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
      ctx.restore();

      if (pose) {
        // Frame-Recording (Benchmark) — Pose im Original (unmirrored) übergeben
        if (onFrameRef.current) {
          onFrameRef.current(pose);
        }

        const mirroredKeypoints = pose.keypoints.map((kp) => ({
          ...kp,
          x: canvas.width - kp.x,
        }));

        drawSkeleton(ctx, mirroredKeypoints);

        const leftHip = mirroredKeypoints[KEYPOINT_INDICES.leftHip] as Keypoint;
        const leftKnee = mirroredKeypoints[
          KEYPOINT_INDICES.leftKnee
        ] as Keypoint;
        const leftAnkle = mirroredKeypoints[
          KEYPOINT_INDICES.leftAnkle
        ] as Keypoint;
        const leftShoulder = mirroredKeypoints[
          KEYPOINT_INDICES.leftShoulder
        ] as Keypoint;
        const leftElbow = mirroredKeypoints[
          KEYPOINT_INDICES.leftElbow
        ] as Keypoint;
        const leftWrist = mirroredKeypoints[
          KEYPOINT_INDICES.leftWrist
        ] as Keypoint;

        const angles: BiomechanicalAngles = {
          knee: calculateKneeAngle(leftHip, leftKnee, leftAnkle),
          hip: calculateHipAngle(leftShoulder, leftHip, leftKnee),
          ankle: calculateAnkleAngle(leftKnee, leftAnkle),
          elbow: calculateElbowAngle(leftShoulder, leftElbow, leftWrist),
          back: calculateBackAngle(leftShoulder, leftHip),
        };

        const relevantKeypoints = [
          leftHip,
          leftKnee,
          leftAnkle,
          leftShoulder,
          leftElbow,
        ];
        const avgConfidence =
          relevantKeypoints.reduce((sum, kp) => sum + (kp.score || 0), 0) /
          relevantKeypoints.length;

        analyzer.addFrame(angles, avgConfidence);
        setCurrentAngles(angles);
        setPoseConfidence(avgConfidence);
        setCycleCount(analyzer.getCycleCount());

        if (angles.knee !== null) {
          drawAngleOverlay(
            ctx,
            { x: leftKnee.x + 15, y: leftKnee.y },
            angles.knee,
            'Knie'
          );
        }
        if (angles.hip !== null) {
          drawAngleOverlay(
            ctx,
            { x: leftHip.x + 15, y: leftHip.y },
            angles.hip,
            'Hüfte'
          );
        }
        if (angles.back !== null) {
          drawAngleOverlay(
            ctx,
            { x: leftShoulder.x + 15, y: leftShoulder.y - 30 },
            angles.back,
            'Rücken'
          );
        }

        if (analyzer.isComplete()) {
          setPhase('complete');
          const results = analyzer.getResults();
          if (results) {
            onComplete(results);
          }
          return;
        }
      }

      animationRef.current = requestAnimationFrame(renderLoop);
    };

    animationRef.current = requestAnimationFrame(renderLoop);

    return () => {
      cancelAnimationFrame(animationRef.current);
    };
  }, [
    isDetectorReady,
    detectPose,
    phase,
    drawSkeleton,
    drawAngleOverlay,
    onComplete,
  ]);

  // Setup-Loop (Live-Preview ohne Analyse)
  useEffect(() => {
    if (!isDetectorReady || phase !== 'setup') return;

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
      ctx.scale(-1, 1);
      ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
      ctx.restore();

      if (pose) {
        const mirroredKeypoints = pose.keypoints.map((kp) => ({
          ...kp,
          x: canvas.width - kp.x,
        }));
        drawSkeleton(ctx, mirroredKeypoints);

        const relevantKeypoints = [
          mirroredKeypoints[KEYPOINT_INDICES.leftHip],
          mirroredKeypoints[KEYPOINT_INDICES.leftKnee],
          mirroredKeypoints[KEYPOINT_INDICES.leftAnkle],
          mirroredKeypoints[KEYPOINT_INDICES.leftShoulder],
        ];
        const avgConfidence =
          relevantKeypoints.reduce((sum, kp) => sum + (kp.score || 0), 0) /
          relevantKeypoints.length;
        setPoseConfidence(avgConfidence);
      }

      animationRef.current = requestAnimationFrame(renderLoop);
    };

    animationRef.current = requestAnimationFrame(renderLoop);

    return () => {
      cancelAnimationFrame(animationRef.current);
    };
  }, [isDetectorReady, detectPose, phase, drawSkeleton]);

  const handleStartAnalysis = () => {
    if (analyzerRef.current) {
      analyzerRef.current.reset();
    }
    setCycleCount(0);
    setPhase('recording');
  };

  const handleStopAnalysis = () => {
    setPhase('setup');
    if (analyzerRef.current) {
      analyzerRef.current.reset();
    }
    setCycleCount(0);
  };

  return (
    <div className="space-y-4">
      <div className="relative w-full max-w-4xl mx-auto">
        <video ref={videoRef} className="hidden" playsInline muted />
        <canvas
          ref={canvasRef}
          className="w-full h-auto rounded-lg shadow-2xl border-2 border-gray-700"
        />

        <div className="absolute top-4 left-4 right-4 flex justify-between items-start">
          {phase === 'recording' && (
            <div className="bg-black/80 backdrop-blur-sm rounded-lg px-4 py-3">
              <div className="text-sm text-gray-400 mb-1">Aufnahme läuft</div>
              <div className="flex items-center gap-3">
                <div className="text-2xl font-bold text-green-400">
                  {cycleCount} / {targetCycles}
                </div>
                <div className="text-sm text-gray-400">Zyklen</div>
              </div>
              <div className="mt-2 h-2 bg-gray-700 rounded-full overflow-hidden w-48">
                <div
                  className="h-full bg-green-400 transition-all duration-300"
                  style={{ width: `${(cycleCount / targetCycles) * 100}%` }}
                />
              </div>
            </div>
          )}

          <div className="bg-black/80 backdrop-blur-sm rounded-lg px-4 py-3">
            <div className="text-sm text-gray-400 mb-1">Pose-Qualität</div>
            <div className="flex items-center gap-2">
              <div
                className={`w-3 h-3 rounded-full ${
                  poseConfidence > 0.7
                    ? 'bg-green-400'
                    : poseConfidence > 0.5
                    ? 'bg-yellow-400'
                    : 'bg-red-400'
                }`}
              />
              <span className="font-mono text-lg">
                {(poseConfidence * 100).toFixed(0)}%
              </span>
            </div>
          </div>
        </div>

        {!isDetectorReady && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/70 rounded-lg">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-green-400 mx-auto mb-4" />
              <p className="text-lg">MoveNet (TFLite) wird geladen...</p>
            </div>
          </div>
        )}
      </div>

      {phase === 'recording' && (
        <div className="grid grid-cols-5 gap-2 max-w-4xl mx-auto">
          <AngleDisplay label="Knie" value={currentAngles.knee} />
          <AngleDisplay label="Hüfte" value={currentAngles.hip} />
          <AngleDisplay label="Rücken" value={currentAngles.back} />
          <AngleDisplay label="Ellbogen" value={currentAngles.elbow} />
          <AngleDisplay label="Knöchel" value={currentAngles.ankle} />
        </div>
      )}

      {phase === 'setup' && (
        <div className="max-w-4xl mx-auto bg-gray-800/50 rounded-lg p-6 border border-gray-700">
          <h3 className="font-semibold text-green-400 mb-4 text-lg">
            Vorbereitung
          </h3>
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-medium text-white mb-2">Kamera-Position</h4>
              <ul className="text-sm text-gray-400 space-y-2">
                <li className="flex items-start gap-2">
                  <span className="text-green-400">1.</span>
                  <span>Positioniere die Kamera seitlich zum Fahrrad</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-400">2.</span>
                  <span>Kamera auf Höhe des Tretlagers ausrichten</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-400">3.</span>
                  <span>Abstand: 2-3 Meter für beste Ergebnisse</span>
                </li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium text-white mb-2">Körper-Position</h4>
              <ul className="text-sm text-gray-400 space-y-2">
                <li className="flex items-start gap-2">
                  <span className="text-green-400">1.</span>
                  <span>
                    Schulter, Hüfte, Knie und Knöchel müssen sichtbar sein
                  </span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-400">2.</span>
                  <span>Tritt gleichmäßig mit normaler Kadenz</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-green-400">3.</span>
                  <span>Vermeide Bewegungen außerhalb der Pedaldrehung</span>
                </li>
              </ul>
            </div>
          </div>

          <div className="mt-6 flex flex-col sm:flex-row gap-3">
            <button
              onClick={handleStartAnalysis}
              disabled={!isDetectorReady || poseConfidence < 0.3}
              className={`flex-1 py-3 px-6 rounded-lg font-semibold transition-all ${
                isDetectorReady && poseConfidence >= 0.3
                  ? 'bg-green-500 hover:bg-green-600 text-white'
                  : 'bg-gray-600 text-gray-400 cursor-not-allowed'
              }`}
            >
              {poseConfidence < 0.3 ? 'Pose nicht erkannt' : 'Analyse starten'}
            </button>
            <button
              onClick={onCancel}
              className="py-3 px-6 rounded-lg font-semibold bg-gray-700 hover:bg-gray-600 text-gray-300 transition-all"
            >
              Abbrechen
            </button>
          </div>
        </div>
      )}

      {phase === 'recording' && (
        <div className="max-w-4xl mx-auto flex justify-center">
          <button
            onClick={handleStopAnalysis}
            className="py-3 px-8 rounded-lg font-semibold bg-red-500 hover:bg-red-600 text-white transition-all"
          >
            Aufnahme abbrechen
          </button>
        </div>
      )}
    </div>
  );
}

interface AngleDisplayProps {
  label: string;
  value: number | null;
}

function AngleDisplay({ label, value }: AngleDisplayProps) {
  return (
    <div className="bg-gray-800/80 rounded-lg p-3 text-center">
      <div className="text-xs text-gray-400 mb-1">{label}</div>
      <div
        className="text-lg font-mono font-bold"
        style={{ color: value !== null ? getAngleColor(value) : '#6b7280' }}
      >
        {value !== null ? `${value.toFixed(0)}°` : '--'}
      </div>
    </div>
  );
}
