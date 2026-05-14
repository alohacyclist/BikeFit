import { useRef, useEffect, useCallback } from 'react';
import * as poseDetection from '@tensorflow-models/pose-detection';
import {
  calculateKneeAngle,
  calculateHipAngle,
  getAngleColor,
  type Keypoint
} from '../utils/AngleCalculator';

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
  rightAnkle: 16
};

// Skeleton-Verbindungen für das Zeichnen
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
  [KEYPOINT_INDICES.rightKnee, KEYPOINT_INDICES.rightAnkle]
];

interface VideoCanvasProps {
  poses: poseDetection.Pose[];
  onDetect: (video: HTMLVideoElement) => Promise<void>;
  isDetectorReady: boolean;
  onAnglesUpdate: (kneeAngle: number | null, hipAngle: number | null) => void;
}

export function VideoCanvas({
  poses,
  onDetect,
  isDetectorReady,
  onAnglesUpdate
}: VideoCanvasProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>(0);

  // Webcam initialisieren
  useEffect(() => {
    async function setupCamera() {
      if (!videoRef.current) return;

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            frameRate: { ideal: 30, max: 60 }
          },
          audio: false
        });

        videoRef.current.srcObject = stream;

        // Warten bis Video bereit ist
        await new Promise<void>((resolve) => {
          if (videoRef.current) {
            videoRef.current.onloadedmetadata = () => {
              resolve();
            };
          }
        });

        await videoRef.current.play();
        console.log('Webcam gestartet:', videoRef.current.videoWidth, 'x', videoRef.current.videoHeight);
      } catch (err) {
        console.error('Webcam Zugriff fehlgeschlagen:', err);
      }
    }

    setupCamera();

    return () => {
      // Cleanup: Stream stoppen
      if (videoRef.current?.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  // Skeleton und Keypoints zeichnen
  const drawSkeleton = useCallback((
    ctx: CanvasRenderingContext2D,
    keypoints: poseDetection.Keypoint[],
    minConfidence: number = 0.3
  ) => {
    // Skeleton-Linien zeichnen
    ctx.strokeStyle = '#00ff88';
    ctx.lineWidth = 3;

    for (const [startIdx, endIdx] of SKELETON_CONNECTIONS) {
      const start = keypoints[startIdx];
      const end = keypoints[endIdx];

      if (
        start.score !== undefined && start.score > minConfidence &&
        end.score !== undefined && end.score > minConfidence
      ) {
        ctx.beginPath();
        ctx.moveTo(start.x, start.y);
        ctx.lineTo(end.x, end.y);
        ctx.stroke();
      }
    }

    // Keypoints zeichnen
    for (const keypoint of keypoints) {
      if (keypoint.score !== undefined && keypoint.score > minConfidence) {
        ctx.fillStyle = '#ff0066';
        ctx.beginPath();
        ctx.arc(keypoint.x, keypoint.y, 6, 0, 2 * Math.PI);
        ctx.fill();

        // Weißer Rand
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }
  }, []);

  // Winkel am Gelenk anzeigen
  const drawAngle = useCallback((
    ctx: CanvasRenderingContext2D,
    position: { x: number; y: number },
    angle: number,
    label: string
  ) => {
    const color = getAngleColor(angle);
    const text = `${label}: ${angle.toFixed(1)}°`;

    // Hintergrund
    ctx.font = 'bold 16px monospace';
    const textMetrics = ctx.measureText(text);
    const padding = 6;

    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.fillRect(
      position.x - padding,
      position.y - 20 - padding,
      textMetrics.width + padding * 2,
      24 + padding
    );

    // Text
    ctx.fillStyle = color;
    ctx.fillText(text, position.x, position.y - 6);
  }, []);

  // Hauptrender-Loop
  useEffect(() => {
    if (!isDetectorReady) return;

    const renderLoop = async () => {
      const video = videoRef.current;
      const canvas = canvasRef.current;

      if (!video || !canvas || video.readyState < 2) {
        animationRef.current = requestAnimationFrame(renderLoop);
        return;
      }

      // Canvas-Größe anpassen
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }

      const ctx = canvas.getContext('2d');
      if (!ctx) {
        animationRef.current = requestAnimationFrame(renderLoop);
        return;
      }

      // Pose Detection durchführen
      await onDetect(video);

      // Video-Frame zeichnen (gespiegelt für natürlichere Ansicht)
      ctx.save();
      ctx.scale(-1, 1);
      ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
      ctx.restore();

      // Poses verarbeiten und zeichnen
      if (poses.length > 0) {
        const pose = poses[0];
        const keypoints = pose.keypoints;

        // Keypoints für gespiegeltes Video anpassen
        const mirroredKeypoints = keypoints.map(kp => ({
          ...kp,
          x: canvas.width - kp.x
        }));

        // Skeleton zeichnen
        drawSkeleton(ctx, mirroredKeypoints);

        // Winkel berechnen (linke Seite - typisch für Bike-Fitting)
        const leftHip = mirroredKeypoints[KEYPOINT_INDICES.leftHip] as Keypoint;
        const leftKnee = mirroredKeypoints[KEYPOINT_INDICES.leftKnee] as Keypoint;
        const leftAnkle = mirroredKeypoints[KEYPOINT_INDICES.leftAnkle] as Keypoint;
        const leftShoulder = mirroredKeypoints[KEYPOINT_INDICES.leftShoulder] as Keypoint;

        const kneeAngle = calculateKneeAngle(leftHip, leftKnee, leftAnkle);
        const hipAngle = calculateHipAngle(leftShoulder, leftHip, leftKnee);

        // Winkel-Callback
        onAnglesUpdate(kneeAngle, hipAngle);

        // Winkel am Gelenk anzeigen
        if (kneeAngle !== null) {
          drawAngle(ctx, { x: leftKnee.x + 20, y: leftKnee.y }, kneeAngle, 'Knie');
        }

        if (hipAngle !== null) {
          drawAngle(ctx, { x: leftHip.x + 20, y: leftHip.y }, hipAngle, 'Hüfte');
        }
      }

      animationRef.current = requestAnimationFrame(renderLoop);
    };

    animationRef.current = requestAnimationFrame(renderLoop);

    return () => {
      cancelAnimationFrame(animationRef.current);
    };
  }, [isDetectorReady, poses, onDetect, drawSkeleton, drawAngle, onAnglesUpdate]);

  return (
    <div className="relative w-full max-w-4xl mx-auto">
      {/* Hidden Video Element */}
      <video
        ref={videoRef}
        className="hidden"
        playsInline
        muted
      />

      {/* Canvas für Rendering */}
      <canvas
        ref={canvasRef}
        className="w-full h-auto rounded-lg shadow-2xl border-2 border-gray-700"
      />

      {/* Loading Overlay */}
      {!isDetectorReady && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/70 rounded-lg">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-green-400 mx-auto mb-4"></div>
            <p className="text-lg">MoveNet wird geladen...</p>
          </div>
        </div>
      )}
    </div>
  );
}
