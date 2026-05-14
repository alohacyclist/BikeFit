import { useState, useCallback, useEffect, useRef } from 'react';
import { usePoseDetection, type Pose } from './hooks/usePoseDetection';
import { VideoCanvas } from './components/VideoCanvas';
import { MetricsOverlay } from './components/MetricsOverlay';
import { AnalysisView } from './components/AnalysisView';
import { ResultsDashboard } from './components/ResultsDashboard';
import { HistoryView } from './components/HistoryView';
import { QuantizationControls } from './components/QuantizationControls';
import { AnalysisResults } from './services/BiomechanicalAnalyzer';
import { generateRecommendations } from './services/RecommendationsEngine';
import { saveToHistory, HistoryEntry } from './services/HistoryStorage';
import { benchmarkExporter } from './services/BenchmarkExporter';
import { calculateKneeAngle } from './utils/AngleCalculator';
import { KP, type QuantizationLevel } from './types/quantization';

type AppView = 'home' | 'analysis' | 'results' | 'history';

function App() {
  const {
    poses,
    metrics,
    isLoading,
    error,
    detectPose,
    detector,
    loadModel,
    currentLevel,
    isWarmingUp,
    lastMeasurementRef,
  } = usePoseDetection('fp32');

  const [kneeAngle, setKneeAngle] = useState<number | null>(null);
  const [hipAngle, setHipAngle] = useState<number | null>(null);
  const [currentView, setCurrentView] = useState<AppView>('home');
  const [analysisResults, setAnalysisResults] =
    useState<AnalysisResults | null>(null);

  // Benchmark-State
  const [participantId, setParticipantId] = useState('P01');
  const sessionStartedRef = useRef(false);

  // Session beim ersten Detector-Ready starten
  useEffect(() => {
    if (detector && !sessionStartedRef.current) {
      benchmarkExporter.startSession(participantId, currentLevel);
      sessionStartedRef.current = true;
    }
  }, [detector, participantId, currentLevel]);

  const handleAnglesUpdate = useCallback(
    (knee: number | null, hip: number | null) => {
      setKneeAngle(knee);
      setHipAngle(hip);
    },
    []
  );

  const handleAnalysisComplete = useCallback((results: AnalysisResults) => {
    setAnalysisResults(results);
    const recommendations = generateRecommendations(results);
    saveToHistory(results, recommendations);
    setCurrentView('results');
  }, []);

  const handleSelectHistoryEntry = useCallback((entry: HistoryEntry) => {
    setAnalysisResults(entry.results);
    setCurrentView('results');
  }, []);

  const handleStartAnalysis = () => {
    setCurrentView('analysis');
    setAnalysisResults(null);
  };

  const handleNewAnalysis = () => {
    setAnalysisResults(null);
    setCurrentView('analysis');
  };

  const handleBackToHome = () => {
    setCurrentView('home');
    setAnalysisResults(null);
  };

  // Quantisierungsstufe wechseln — neue Benchmark-Session beginnen
  const handleLevelChange = useCallback(
    async (level: QuantizationLevel) => {
      if (level === currentLevel) return;
      benchmarkExporter.reset();
      benchmarkExporter.startSession(participantId, level);
      sessionStartedRef.current = true;
      await loadModel(level);
    },
    [currentLevel, participantId, loadModel]
  );

  const handleParticipantIdChange = useCallback(
    (id: string) => {
      setParticipantId(id);
      // Bei laufender Session: neu starten mit aktualisierter ID
      benchmarkExporter.reset();
      benchmarkExporter.startSession(id, currentLevel);
      sessionStartedRef.current = true;
    },
    [currentLevel]
  );

  const handleExport = useCallback(() => {
    benchmarkExporter.downloadJSON();
  }, []);

  // Pro Frame in der Analyse-Phase: Messdaten an Exporter weiterreichen
  const handleFrameMeasurement = useCallback(
    (pose: Pose) => {
      const m = lastMeasurementRef.current;
      if (!m) return;

      const kp = pose.keypoints;
      const kneeRight = calculateKneeAngle(
        kp[KP.RIGHT_HIP],
        kp[KP.RIGHT_KNEE],
        kp[KP.RIGHT_ANKLE]
      );
      const kneeLeft = calculateKneeAngle(
        kp[KP.LEFT_HIP],
        kp[KP.LEFT_KNEE],
        kp[KP.LEFT_ANKLE]
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
    [lastMeasurementRef]
  );

  const renderContent = () => {
    switch (currentView) {
      case 'analysis':
        return (
          <div className="grid lg:grid-cols-[1fr,320px] gap-6">
            <AnalysisView
              detectPose={detectPose}
              isDetectorReady={!isLoading && detector !== null}
              onComplete={handleAnalysisComplete}
              onCancel={handleBackToHome}
              targetCycles={5}
              onFrameMeasurement={handleFrameMeasurement}
            />
            <div className="space-y-4">
              <QuantizationControls
                currentLevel={currentLevel}
                onLevelChange={handleLevelChange}
                isLoading={isLoading}
                isWarmingUp={isWarmingUp}
                participantId={participantId}
                onParticipantIdChange={handleParticipantIdChange}
                onExport={handleExport}
              />
              <MetricsOverlay
                inferenceTime={metrics.inferenceTime}
                fps={metrics.fps}
                frameCount={metrics.frameCount}
                kneeAngle={kneeAngle}
                hipAngle={hipAngle}
              />
            </div>
          </div>
        );

      case 'results':
        if (!analysisResults) {
          setCurrentView('home');
          return null;
        }
        return (
          <ResultsDashboard
            results={analysisResults}
            onNewAnalysis={handleNewAnalysis}
            onShowHistory={() => setCurrentView('history')}
          />
        );

      case 'history':
        return (
          <HistoryView
            onSelectEntry={handleSelectHistoryEntry}
            onBack={handleBackToHome}
            onNewAnalysis={handleStartAnalysis}
          />
        );

      case 'home':
      default:
        return (
          <div className="grid lg:grid-cols-[1fr,320px] gap-6">
            <div>
              <VideoCanvas
                poses={poses}
                onDetect={async (v) => {
                  await detectPose(v);
                }}
                isDetectorReady={!isLoading && detector !== null}
                onAnglesUpdate={handleAnglesUpdate}
              />

              <div className="mt-4 flex flex-col sm:flex-row gap-3">
                <button
                  onClick={handleStartAnalysis}
                  disabled={!detector}
                  className={`flex-1 py-3 px-6 rounded-lg font-semibold transition-all flex items-center justify-center gap-2 ${
                    detector
                      ? 'bg-green-500 hover:bg-green-600 text-white'
                      : 'bg-gray-600 text-gray-400 cursor-not-allowed'
                  }`}
                >
                  Biomechanische Analyse starten
                </button>
                <button
                  onClick={() => setCurrentView('history')}
                  className="py-3 px-6 rounded-lg font-semibold bg-gray-700 hover:bg-gray-600 text-gray-300 transition-all flex items-center justify-center gap-2"
                >
                  Verlauf
                </button>
              </div>

              <div className="mt-4 p-4 bg-gray-800/50 rounded-lg border border-gray-700">
                <h3 className="font-semibold text-green-400 mb-2">Anleitung</h3>
                <ul className="text-sm text-gray-400 space-y-1">
                  <li>
                    • Positioniere dich seitlich zur Kamera auf dem Fahrrad
                  </li>
                  <li>
                    • Schulter, Hüfte, Knie und Knöchel müssen sichtbar sein
                  </li>
                  <li>
                    • Optimaler Kniewinkel am unteren Totpunkt: 140° - 150°
                  </li>
                  <li>
                    • Grün = optimal, Gelb = akzeptabel, Rot = Anpassung
                  </li>
                </ul>
              </div>
            </div>

            <div className="space-y-4">
              <MetricsOverlay
                inferenceTime={metrics.inferenceTime}
                fps={metrics.fps}
                frameCount={metrics.frameCount}
                kneeAngle={kneeAngle}
                hipAngle={hipAngle}
              />

              <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
                <h3 className="font-semibold text-gray-300 mb-3 text-sm">
                  Technische Details
                </h3>
                <div className="space-y-2 text-xs">
                  <InfoRow label="Modell" value="MoveNet Lightning (TFLite)" />
                  <InfoRow label="Backend" value="WASM" />
                  <InfoRow
                    label="Quantisierung"
                    value={currentLevel.toUpperCase()}
                  />
                  <InfoRow label="Verarbeitung" value="100% lokal" />
                </div>
              </div>

              <div className="bg-green-900/30 rounded-lg p-4 border border-green-700/50">
                <span className="font-semibold text-sm text-green-400">
                  Privacy-First
                </span>
                <p className="text-xs text-gray-400 mt-2">
                  Alle Daten werden lokal im Browser verarbeitet. Kein Upload.
                </p>
              </div>
            </div>
          </div>
        );
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900">
      <header className="bg-gray-900/80 backdrop-blur-sm border-b border-gray-700 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="cursor-pointer" onClick={handleBackToHome}>
              <h1 className="text-2xl font-bold text-white">
                EdgeFit <span className="text-green-400">Pro</span>
              </h1>
              <p className="text-sm text-gray-400">
                MoveNet TFLite • Quantisierungs-Benchmark
              </p>
            </div>
            <div className="flex items-center gap-4">
              {currentView !== 'home' && (
                <button
                  onClick={handleBackToHome}
                  className="text-sm text-gray-400 hover:text-white transition-colors"
                >
                  Home
                </button>
              )}
              <div className="flex items-center gap-2">
                <div
                  className={`w-3 h-3 rounded-full ${
                    detector
                      ? isWarmingUp
                        ? 'bg-yellow-400 animate-pulse'
                        : 'bg-green-400 animate-pulse'
                      : 'bg-yellow-400'
                  }`}
                />
                <span className="text-sm text-gray-400 font-mono">
                  {!detector
                    ? 'Lädt...'
                    : isWarmingUp
                    ? `Warmup (${currentLevel})`
                    : `Bereit (${currentLevel})`}
                </span>
              </div>
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

        {renderContent()}
      </main>

      <footer className="mt-8 border-t border-gray-800 bg-gray-900/50">
        <div className="max-w-7xl mx-auto px-4 py-4 text-center text-xs text-gray-500">
          EdgeFit Pro • Bachelorarbeit FOM • TFLite + React
        </div>
      </footer>
    </div>
  );
}

interface InfoRowProps {
  label: string;
  value: string;
}

function InfoRow({ label, value }: InfoRowProps) {
  return (
    <div className="flex justify-between">
      <span className="text-gray-500">{label}</span>
      <span className="text-gray-300 font-mono">{value}</span>
    </div>
  );
}

export default App;
