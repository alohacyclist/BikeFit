import { useState, useCallback, useEffect } from "react";
import { usePoseDetection, type Pose } from "./hooks/usePoseDetection";
import { AnalysisView } from "./components/AnalysisView";
import { QuantizationControls } from "./components/QuantizationControls";
import { VideoSourcePanel } from "./components/VideoSourcePanel";
import { benchmarkExporter } from "./services/BenchmarkExporter";
import { calculateKneeAngle, type BodySide } from "./utils/AngleCalculator";
import { KP, type QuantizationLevel } from "./types/quantization";

function App() {
  const {
    metrics,
    isLoading,
    error,
    detectPose,
    detector,
    loadModel,
    resetBackend,
    beginWarmup,
    currentLevel,
    isWarmingUp,
    lastMeasurementRef,
    modelFingerprint,
    threadingPreference,
    setThreadingPreference,
    multiThreadingAvailable,
    activeThreadingMode,
    activeNumThreads,
  } = usePoseDetection("fp32");

  const [participantId, setParticipantId] = useState("P01");
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [targetFps, setTargetFps] = useState(30);
  const [forcedSide, setForcedSide] = useState<BodySide>("right");
  const [isRecording, setIsRecording] = useState(false);
  const [lastSummary, setLastSummary] = useState<{
    durationSeconds: number;
    expectedFrames: number;
  } | null>(null);

  // Session bei Detektor-Ready / Level- / Side-Wechsel neu starten
  useEffect(() => {
    if (!detector) return;
    benchmarkExporter.reset();
    benchmarkExporter.startSession(participantId, currentLevel);
    benchmarkExporter.setLockedSide(forcedSide);
    benchmarkExporter.setModelFingerprint(modelFingerprint);
    benchmarkExporter.setThreadingMode(activeThreadingMode);
    benchmarkExporter.setNumThreads(activeNumThreads);
    benchmarkExporter.setVideoSource(videoFile?.name, targetFps);
  }, [
    detector,
    currentLevel,
    participantId,
    forcedSide,
    modelFingerprint,
    activeThreadingMode,
    activeNumThreads,
    targetFps,
    videoFile,
  ]);

  const handleLevelChange = useCallback(
    async (level: QuantizationLevel) => {
      if (level === currentLevel) return;
      await resetBackend();
      await loadModel(level);
    },
    [currentLevel, resetBackend, loadModel],
  );

  const handleThreadingChange = useCallback(
    (mode: "single" | "multi") => {
      // setThreadingPreference persistiert in sessionStorage und triggert
      // window.location.reload() — TFLite-UMD ist Page-Singleton, Hot-Swap
      // des Pthread-Pools würde deadlocken.
      setThreadingPreference(mode);
    },
    [setThreadingPreference],
  );

  const handleExport = useCallback(() => {
    benchmarkExporter.downloadJSON();
  }, []);

  const handleSideLocked = useCallback((side: BodySide) => {
    benchmarkExporter.setLockedSide(side);
  }, []);

  const handleRecordingFinalize = useCallback(
    (durationSeconds: number, expectedFrames: number) => {
      benchmarkExporter.setVideoMeta(durationSeconds, expectedFrames);
      setLastSummary({ durationSeconds, expectedFrames });
    },
    [],
  );

  const handleFrameMeasurement = useCallback(
    (pose: Pose) => {
      const m = lastMeasurementRef.current;
      if (!m) return;
      const kp = pose.keypoints;
      const kneeRight = calculateKneeAngle(
        kp[KP.RIGHT_HIP],
        kp[KP.RIGHT_KNEE],
        kp[KP.RIGHT_ANKLE],
      );
      const kneeLeft = calculateKneeAngle(
        kp[KP.LEFT_HIP],
        kp[KP.LEFT_KNEE],
        kp[KP.LEFT_ANKLE],
      );
      benchmarkExporter.recordFrame({
        frameIndex: m.frameIndex,
        timestampMs: performance.now(),
        inferenceMs: m.inferenceMs,
        fps: m.fps,
        kneeAngleRight: kneeRight,
        kneeAngleLeft: kneeLeft,
        keypointScores: kp.map((k) => k.score ?? 0),
        isWarmup: m.isWarmup,
      });
    },
    [lastMeasurementRef],
  );

  const handleResetSession = useCallback(() => {
    setLastSummary(null);
    benchmarkExporter.reset();
    benchmarkExporter.startSession(participantId, currentLevel);
    benchmarkExporter.setLockedSide(forcedSide);
    benchmarkExporter.setModelFingerprint(modelFingerprint);
    benchmarkExporter.setThreadingMode(activeThreadingMode);
    benchmarkExporter.setNumThreads(activeNumThreads);
    benchmarkExporter.setVideoSource(videoFile?.name, targetFps);
  }, [
    participantId,
    currentLevel,
    forcedSide,
    modelFingerprint,
    activeThreadingMode,
    activeNumThreads,
    targetFps,
    videoFile,
  ]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900">
      <header className="bg-gray-900/80 backdrop-blur-sm border-b border-gray-700 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-white">
                EdgeFit <span className="text-green-400">Pro</span>
              </h1>
              <p className="text-sm text-gray-400">
                Kniewinkel-Messung · Quantisierungs-Benchmark
              </p>
            </div>
            <div className="flex items-center gap-2">
              <div
                className={`w-3 h-3 rounded-full ${
                  detector
                    ? isWarmingUp
                      ? "bg-yellow-400 animate-pulse"
                      : "bg-green-400 animate-pulse"
                    : "bg-yellow-400"
                }`}
              />
              <span className="text-sm text-gray-400 font-mono">
                {!detector
                  ? "Lädt..."
                  : isWarmingUp
                    ? `Warmup (${currentLevel})`
                    : `Bereit (${currentLevel}/${activeThreadingMode})`}
              </span>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-6">
        {error && (
          <div className="mb-6 p-4 bg-red-900/50 border border-red-500 rounded-lg text-red-200">
            <strong>Fehler:</strong> {error}
          </div>
        )}

        <div className="grid lg:grid-cols-[1fr,340px] gap-6">
          <AnalysisView
            detectPose={detectPose}
            isDetectorReady={!isLoading && detector !== null}
            onCancel={handleResetSession}
            onFrameMeasurement={handleFrameMeasurement}
            onSideLocked={handleSideLocked}
            onRecordingFinalize={handleRecordingFinalize}
            onRecordingStart={beginWarmup}
            onRecordingActiveChange={setIsRecording}
            onRecordingAbort={handleResetSession}
            videoFile={videoFile}
            forcedSide={forcedSide}
            targetFps={targetFps}
          />

          <div className="space-y-4">
            <VideoSourcePanel
              file={videoFile}
              onFileChange={setVideoFile}
              targetFps={targetFps}
              onTargetFpsChange={setTargetFps}
              disabled={isRecording}
            />

            <QuantizationControls
              currentLevel={currentLevel}
              onLevelChange={handleLevelChange}
              isLoading={isLoading}
              isWarmingUp={isWarmingUp}
              participantId={participantId}
              onParticipantIdChange={setParticipantId}
              onExport={handleExport}
              forcedSide={forcedSide}
              onForcedSideChange={setForcedSide}
              threadingPreference={threadingPreference}
              onThreadingPreferenceChange={handleThreadingChange}
              multiThreadingAvailable={multiThreadingAvailable}
              activeThreadingMode={activeThreadingMode}
              recording={isRecording}
            />

            <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700 text-xs space-y-1">
              <div className="flex justify-between">
                <span className="text-gray-500">Inferenz</span>
                <span className="text-gray-300 font-mono">
                  {metrics.inferenceTime.toFixed(1)} ms
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">FPS</span>
                <span className="text-gray-300 font-mono">
                  {metrics.fps.toFixed(1)}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-500">Frames</span>
                <span className="text-gray-300 font-mono">
                  {metrics.frameCount}
                </span>
              </div>
              {modelFingerprint && (
                <div className="pt-2 border-t border-gray-700">
                  <div className="flex justify-between text-gray-500">
                    <span>SHA-256</span>
                    <span className="font-mono text-gray-400">
                      {modelFingerprint.sha256.substring(0, 12)}…
                    </span>
                  </div>
                  <div className="flex justify-between text-gray-500">
                    <span>Bytes</span>
                    <span className="font-mono">
                      {(modelFingerprint.sizeBytes / 1024 / 1024).toFixed(2)} MB
                    </span>
                  </div>
                </div>
              )}
            </div>

            {lastSummary && (
              <div className="bg-gray-800/50 rounded-lg p-4 border border-green-700/40 text-xs space-y-1">
                <div className="text-sm font-semibold text-green-400 mb-1">
                  Aufnahme abgeschlossen
                </div>
                <div className="flex justify-between text-gray-400">
                  <span>Frames</span>
                  <span className="font-mono">
                    {lastSummary.expectedFrames}
                  </span>
                </div>
                <div className="flex justify-between text-gray-400">
                  <span>Dauer</span>
                  <span className="font-mono">
                    {lastSummary.durationSeconds.toFixed(1)} s
                  </span>
                </div>
                <p className="text-[10px] text-gray-500 pt-1">
                  Auswertung (Zyklen, Statistik, Genauigkeit): JSON exportieren
                  → Python-Postprocessing.
                </p>
                <button
                  onClick={handleResetSession}
                  className="mt-2 w-full py-1.5 px-3 rounded-md text-xs bg-gray-700 hover:bg-gray-600 text-gray-200"
                >
                  Neue Aufnahme
                </button>
              </div>
            )}
          </div>
        </div>
      </main>

      <footer className="mt-8 border-t border-gray-800 bg-gray-900/50">
        <div className="max-w-7xl mx-auto px-4 py-4 text-center text-xs text-gray-500">
          EdgeFit Pro · Bachelorarbeit FOM · TFLite + React
        </div>
      </footer>
    </div>
  );
}

export default App;
