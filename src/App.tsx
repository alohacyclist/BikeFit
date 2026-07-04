import { useState, useCallback, useEffect, useRef } from "react";
import { usePoseDetection, type Pose } from "./hooks/usePoseDetection";
import { AnalysisView } from "./components/AnalysisView";
import { QuantizationControls } from "./components/QuantizationControls";
import { VideoSourcePanel } from "./components/VideoSourcePanel";
import {
  benchmarkExporter,
  type DroppedFrameReason,
} from "./services/BenchmarkExporter";
import { calculateKneeAngle, type BodySide } from "./utils/AngleCalculator";
import { detectHardware } from "./utils/hardwareInfo";
import * as videoStore from "./utils/videoStore";
import {
  readPending,
  writePending,
  clearPending,
  type PendingBoot,
} from "./utils/sequenceResume";
import {
  KP,
  TFLITE_MODEL_URLS,
  type QuantizationLevel,
} from "./types/quantization";

const HW_DEVICE_KEY = "edgefit.hw.device";
const HW_CPU_KEY = "edgefit.hw.cpu";
const PID_KEY = "edgefit.probandId";
const SIDE_KEY = "edgefit.bodySide";
const FPS_KEY = "edgefit.targetFps";

// Ganze Sequenz. Der Wechsel INS/AUS int8 deadlockt bei In-Page-Modellwechsel
// (qu8-Delegate-Swap auf dem page-lifetime-Singleton-Pthread-Pool). Deshalb
// läuft JEDER Stufenwechsel über einen Reload → frischer Pool, kein Swap.
const SEQUENCE_ORDER: QuantizationLevel[] = ["fp32", "fp16", "int8"];

const delay = (ms: number) => new Promise((r) => setTimeout(r, ms));

function App() {
  // Beim Boot ausstehenden Stufen-/Sequenz-Zustand EINMAL lesen — bestimmt die
  // initial zu ladende Stufe (statt immer fp32) und ob automatisch aufgenommen
  // wird (Voll-Sequenz-Resume).
  const bootRef = useRef<PendingBoot | null>(readPending());
  const boot = bootRef.current;
  const initialLevel: QuantizationLevel = boot?.level ?? "fp32";

  const {
    metrics,
    isLoading,
    error,
    detectPose,
    detector,
    beginWarmup,
    currentLevel,
    isWarmingUp,
    lastMeasurementRef,
    modelLoadMsRef,
    modelFingerprint,
    threadingPreference,
    setThreadingPreference,
    multiThreadingAvailable,
    activeThreadingMode,
  } = usePoseDetection(initialLevel);

  const [participantId, setParticipantId] = useState(
    () => localStorage.getItem(PID_KEY) ?? "P01",
  );
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [targetFps, setTargetFps] = useState(
    () => Number(localStorage.getItem(FPS_KEY)) || 30,
  );
  const [forcedSide, setForcedSide] = useState<BodySide>(() =>
    localStorage.getItem(SIDE_KEY) === "right" ? "right" : "left",
  );
  const [isRecording, setIsRecording] = useState(false);
  const [runIndex, setRunIndex] = useState(0);
  const [device, setDevice] = useState(
    () => localStorage.getItem(HW_DEVICE_KEY) ?? "",
  );
  const [cpu, setCpu] = useState(() => localStorage.getItem(HW_CPU_KEY) ?? "");
  const [lastSummary, setLastSummary] = useState<{
    durationSeconds: number;
    expectedFrames: number;
  } | null>(null);
  // In IndexedDB gesammelte Sequenz-JSONs, die auf einen gestenbasierten
  // Download warten (Auto-Download nach Reload wird von Chrome blockiert).
  const [pendingResults, setPendingResults] = useState<
    videoStore.StoredResult[]
  >([]);

  // Voll-Sequenz-Zustand (überlebt Reloads via sessionStorage/IndexedDB).
  const [sequenceActive, setSequenceActive] = useState<boolean>(
    boot?.autoRecord ?? false,
  );
  const remainingRef = useRef<QuantizationLevel[]>(boot?.remaining ?? []);
  const [autoStartTrigger, setAutoStartTrigger] = useState(0);
  const autoStartedRef = useRef(false);

  // --- Persistente Setup-Setter (überleben Reload via localStorage) ---
  const persist = (key: string, value: string) => {
    try {
      localStorage.setItem(key, value);
    } catch {
      // ignore
    }
  };
  const changeParticipantId = useCallback((v: string) => {
    setParticipantId(v);
    persist(PID_KEY, v);
  }, []);
  const changeForcedSide = useCallback((v: BodySide) => {
    setForcedSide(v);
    persist(SIDE_KEY, v);
  }, []);
  const changeTargetFps = useCallback((v: number) => {
    setTargetFps(v);
    persist(FPS_KEY, String(v));
  }, []);
  const changeDevice = useCallback((v: string) => {
    setDevice(v);
    persist(HW_DEVICE_KEY, v);
  }, []);
  const changeCpu = useCallback((v: string) => {
    setCpu(v);
    persist(HW_CPU_KEY, v);
  }, []);

  // runIndex je (Proband × Threading) persistieren.
  const runKey = `edgefit.runIndex.${participantId}.${threadingPreference}`;
  const runKeyRef = useRef(runKey);
  useEffect(() => {
    runKeyRef.current = runKey;
  }, [runKey]);
  useEffect(() => {
    try {
      const stored = localStorage.getItem(runKey);
      setRunIndex(stored !== null ? Number(stored) || 0 : 0);
    } catch {
      setRunIndex(0);
    }
  }, [runKey]);
  const setRunIndexPersist = useCallback((value: number) => {
    const v = Number.isFinite(value) && value >= 0 ? Math.floor(value) : 0;
    setRunIndex(v);
    persist(runKeyRef.current, String(v));
  }, []);
  const bumpRunIndex = useCallback(() => {
    setRunIndex((r) => {
      const next = r + 1;
      persist(runKeyRef.current, String(next));
      return next;
    });
  }, []);

  // --- Boot-Resume: Video wiederherstellen, Pending konsumieren ---
  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (boot) {
        // Video aus IndexedDB zurückholen (Reload verwarf das File-Objekt).
        try {
          const file = await videoStore.loadVideo();
          if (!cancelled && file) setVideoFile(file);
        } catch {
          // ignore
        }
        // Einzelwechsel (kein autoRecord): Pending ist Einmal-Direktive.
        if (!boot.autoRecord) clearPending();
      } else {
        // Frischer Start: evtl. Alt-Video aus einer früheren Sitzung verwerfen.
        try {
          await videoStore.clearVideo();
        } catch {
          // ignore
        }
      }
      // Nicht mitten in einer Sequenz: evtl. noch nicht heruntergeladene
      // Ergebnis-JSONs anbieten (überleben Reload/Tab-Schließen in IndexedDB).
      if (!boot?.autoRecord) {
        try {
          const results = await videoStore.loadResults();
          if (!cancelled && results.length) setPendingResults(results);
        } catch {
          // ignore
        }
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Zentrale Session-Initialisierung nach Vertrag v1.0.0.
  const initSession = useCallback(() => {
    benchmarkExporter.reset();
    benchmarkExporter.startSession(participantId, currentLevel, runIndex);
    benchmarkExporter.setBodySide(forcedSide);
    benchmarkExporter.setThreading(
      activeThreadingMode === "multi" ? "multi" : "single",
    );
    benchmarkExporter.setModel(
      modelFingerprint?.sha256 ?? "",
      modelFingerprint?.url ?? TFLITE_MODEL_URLS[currentLevel],
    );
    benchmarkExporter.setTargetFps(targetFps);
    const hw = detectHardware();
    benchmarkExporter.setHardware({
      ...hw,
      device: device.trim() || hw.device,
      cpu: cpu.trim() || hw.cpu,
    });
  }, [
    participantId,
    currentLevel,
    runIndex,
    forcedSide,
    activeThreadingMode,
    modelFingerprint,
    targetFps,
    device,
    cpu,
  ]);

  useEffect(() => {
    if (!detector) return;
    initSession();
  }, [detector, initSession]);

  // Auto-Aufnahme im Sequenz-Boot: einmal triggern, sobald Modell bereit.
  // AnalysisView startet dann, sobald zusätzlich die Videoquelle bereit ist.
  useEffect(() => {
    if (!sequenceActive || autoStartedRef.current) return;
    if (detector && !isLoading && !isWarmingUp) {
      autoStartedRef.current = true;
      setAutoStartTrigger((n) => n + 1);
    }
  }, [sequenceActive, detector, isLoading, isWarmingUp]);

  // --- Reload-getriebener Stufenwechsel (frischer Pthread-Pool) ---
  const goToLevel = useCallback(
    async (
      level: QuantizationLevel,
      remaining: QuantizationLevel[],
      autoRecord: boolean,
    ) => {
      // Video über den Reload retten.
      try {
        if (videoFile) await videoStore.saveVideo(videoFile);
      } catch {
        // ignore — ohne Video kann eine Sequenz nicht fortsetzen
      }
      writePending({ level, remaining, autoRecord });
      window.location.reload();
    },
    [videoFile],
  );

  const handleLevelChange = useCallback(
    async (level: QuantizationLevel) => {
      if (level === currentLevel || isRecording) return;
      await goToLevel(level, [], false);
    },
    [currentLevel, isRecording, goToLevel],
  );

  const handleThreadingChange = useCallback(
    (mode: "single" | "multi") => {
      setThreadingPreference(mode);
    },
    [setThreadingPreference],
  );

  const handleExport = useCallback(() => {
    benchmarkExporter.downloadJSON();
  }, []);

  const handleSideLocked = useCallback((side: BodySide) => {
    benchmarkExporter.setBodySide(side);
  }, []);

  const handleRunFullSequence = useCallback(async () => {
    if (!videoFile || isRecording) return;
    // Ergebnisse einer früheren Sequenz verwerfen, damit nichts vermischt wird.
    try {
      await videoStore.clearResults();
    } catch {
      // ignore
    }
    setPendingResults([]);
    const [first, ...rest] = SEQUENCE_ORDER;
    // Sequenz startet mit einem Reload auf die erste Stufe (frischer Pool),
    // autoRecord treibt danach jede Stufe über Reloads hinweg.
    await goToLevel(first, rest, true);
  }, [videoFile, isRecording, goToLevel]);

  // Alle gesammelten Sequenz-JSONs mit EINER User-Geste herunterladen (umgeht
  // Chromes Blockade automatischer Downloads nach Reload).
  const downloadResults = useCallback(() => {
    for (const r of pendingResults) {
      const blob = new Blob([r.json], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = r.filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(() => URL.revokeObjectURL(url), 2000);
    }
    videoStore.clearResults().catch(() => {});
    setPendingResults([]);
  }, [pendingResults]);

  const handleRecordingFinalize = useCallback(
    async (durationSeconds: number, totalFrames: number) => {
      benchmarkExporter.setVideoMeta(durationSeconds, totalFrames);
      benchmarkExporter.setModelLoadMs(modelLoadMsRef.current ?? 0);
      setLastSummary({ durationSeconds, expectedFrames: totalFrames });

      if (!sequenceActive) {
        // Einzelaufnahme: kein Reload dazwischen → Auto-Download hängt an der
        // Start-Geste und funktioniert.
        benchmarkExporter.downloadJSON();
        return;
      }

      // Voll-Sequenz: NICHT automatisch herunterladen. Nach jedem Reload wäre
      // der Download un-gestured → Chrome blockt ihn (nur die erste Stufe käme
      // durch). Stattdessen JSON in IndexedDB sammeln und am Ende per Button
      // (User-Geste) alle gemeinsam herunterladen.
      try {
        await videoStore.saveResult(
          currentLevel,
          benchmarkExporter.getFilename(),
          benchmarkExporter.exportJSON(),
        );
      } catch {
        // ignore
      }

      const remaining = remainingRef.current;
      if (remaining.length > 0) {
        const [next, ...rest] = remaining;
        await delay(300);
        await goToLevel(next, rest, true);
      } else {
        // Sequenz komplett: runIndex erhöhen, gesammelte JSONs zum Download
        // anbieten.
        bumpRunIndex();
        remainingRef.current = [];
        setSequenceActive(false);
        clearPending();
        try {
          setPendingResults(await videoStore.loadResults());
        } catch {
          // ignore
        }
      }
    },
    [sequenceActive, currentLevel, goToLevel, bumpRunIndex, modelLoadMsRef],
  );

  const handleDroppedFrame = useCallback(
    (frameIndex: number, reason: DroppedFrameReason, message?: string) => {
      benchmarkExporter.recordDropped({ frameIndex, reason, message });
    },
    [],
  );

  const handleFrameMeasurement = useCallback(
    (pose: Pose, frameIndex: number) => {
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
        frameIndex,
        timestampMs: m.timestampMs,
        inferenceMs: m.inferenceMs,
        fps: m.inferenceMs > 0 ? 1000 / m.inferenceMs : 0,
        kneeAngleRight: kneeRight,
        kneeAngleLeft: kneeLeft,
        keypoints: kp.map((k) => ({ x: k.x, y: k.y })),
        keypointScores: kp.map((k) => k.score ?? 0),
        isWarmup: m.isWarmup,
      });
    },
    [lastMeasurementRef],
  );

  const handleResetSession = useCallback(() => {
    // Abbruch beendet auch eine laufende Voll-Sequenz.
    setSequenceActive(false);
    remainingRef.current = [];
    clearPending();
    setLastSummary(null);
    initSession();
  }, [initSession]);

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
            onDroppedFrame={handleDroppedFrame}
            onSideLocked={handleSideLocked}
            onRecordingFinalize={handleRecordingFinalize}
            onRecordingStart={beginWarmup}
            onRecordingActiveChange={setIsRecording}
            onRecordingAbort={handleResetSession}
            videoFile={videoFile}
            forcedSide={forcedSide}
            targetFps={targetFps}
            autoStartTrigger={autoStartTrigger}
            inSequence={sequenceActive}
          />

          <div className="space-y-4">
            <VideoSourcePanel
              file={videoFile}
              onFileChange={setVideoFile}
              targetFps={targetFps}
              onTargetFpsChange={changeTargetFps}
              disabled={isRecording}
            />

            <QuantizationControls
              currentLevel={currentLevel}
              onLevelChange={handleLevelChange}
              isLoading={isLoading}
              isWarmingUp={isWarmingUp}
              participantId={participantId}
              onParticipantIdChange={changeParticipantId}
              runIndex={runIndex}
              onRunIndexChange={setRunIndexPersist}
              device={device}
              onDeviceChange={changeDevice}
              cpu={cpu}
              onCpuChange={changeCpu}
              onExport={handleExport}
              forcedSide={forcedSide}
              onForcedSideChange={changeForcedSide}
              threadingPreference={threadingPreference}
              onThreadingPreferenceChange={handleThreadingChange}
              multiThreadingAvailable={multiThreadingAvailable}
              activeThreadingMode={activeThreadingMode}
              recording={isRecording}
              onRunFullSequence={handleRunFullSequence}
              sequenceTotal={SEQUENCE_ORDER.length}
              sequenceRemaining={
                sequenceActive ? remainingRef.current.length + 1 : 0
              }
              canStartSequence={
                !!videoFile &&
                !!detector &&
                !isLoading &&
                !isWarmingUp &&
                !isRecording &&
                !sequenceActive
              }
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

            {sequenceActive && (
              <div className="bg-blue-900/30 rounded-lg p-3 border border-blue-700/40 text-xs text-blue-200">
                Voll-Sequenz läuft ({currentLevel.toUpperCase()}) — nach jeder
                Stufe lädt die App per Reload neu (frischer WASM-Pool). Die
                JSONs werden gesammelt und am Ende auf Klick heruntergeladen.
              </div>
            )}

            {pendingResults.length > 0 && !sequenceActive && (
              <div className="bg-green-900/30 rounded-lg p-4 border border-green-600/50 space-y-2">
                <div className="text-sm font-semibold text-green-300">
                  Sequenz fertig — {pendingResults.length} JSON bereit
                </div>
                <p className="text-[11px] text-gray-300 leading-snug">
                  {pendingResults.map((r) => r.level).join(", ")}. Ein Klick
                  lädt alle. Chrome fragt evtl. „mehrere Dateien herunterladen?"
                  → Zulassen.
                </p>
                <button
                  onClick={downloadResults}
                  className="w-full py-2.5 px-3 rounded-md text-sm font-bold bg-green-500 hover:bg-green-600 text-white"
                >
                  {pendingResults.length} JSON herunterladen
                </button>
              </div>
            )}

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
