/**
 * Umgebungs-/Reproduzierbarkeits-Erfassung pro Benchmark-Session (Vertrag v1.1.0).
 *
 * Sammelt browser-lesbare Laufzeitfakten, die für die spätere Analyse von
 * Reihenfolge-/Thermik-Effekten (Latenz vs. Zeit) nötig sind: tatsächlich
 * genutzte numThreads, crossOriginIsolated, tfjs-Versionen, macOS-Version
 * (via userAgentData, umgeht den UA-Freeze), Batteriestand.
 *
 * NICHT aus JavaScript lesbar und daher manuell (UI-Toggles → power-Block):
 * AC-Netzbetrieb, „Low Power Mode aus", exakte macOS-Build-Nummer. Thermisches
 * Throttling wird parallel via `pmset -g thermlog` erfasst und über capturedAt
 * korreliert.
 */

import pkg from "../../package.json";

const deps = (pkg as { dependencies: Record<string, string> }).dependencies;

export interface TfjsVersions {
  core: string;
  backendWasm: string;
  backendWebgl: string;
  converter: string;
  tflite: string;
  poseDetection: string;
}

export interface UserAgentDataInfo {
  platform: string | null;
  platformVersion: string | null;
  uaFullVersion: string | null;
  model: string | null;
  architecture: string | null;
  bitness: string | null;
}

export interface BatteryInfo {
  level: number | null; // 0..1
  charging: boolean | null;
}

/** Manuell erfasste Werte (nicht JS-lesbar) — vom Studienleiter per UI gesetzt. */
export interface ManualPowerInfo {
  acPower: boolean;
  lowPowerModeOff: boolean;
  batteryPercentNote: number | null;
}

export interface EnvironmentInfo {
  capturedAt: string;
  crossOriginIsolated: boolean;
  hardwareConcurrency: number;
  deviceMemoryGb: number | null;
  numThreads: number | null;
  threadingMode: "single" | "multi";
  tfjs: TfjsVersions;
  userAgentData: UserAgentDataInfo | null;
  battery: BatteryInfo | null;
  power: ManualPowerInfo;
}

const TFJS_VERSIONS: TfjsVersions = {
  core: deps["@tensorflow/tfjs-core"] ?? "unknown",
  backendWasm: deps["@tensorflow/tfjs-backend-wasm"] ?? "unknown",
  backendWebgl: deps["@tensorflow/tfjs-backend-webgl"] ?? "unknown",
  converter: deps["@tensorflow/tfjs-converter"] ?? "unknown",
  tflite: deps["@tensorflow/tfjs-tflite"] ?? "unknown",
  poseDetection: deps["@tensorflow-models/pose-detection"] ?? "unknown",
};

interface UADataLike {
  getHighEntropyValues?: (hints: string[]) => Promise<Record<string, unknown>>;
}

async function readUserAgentData(): Promise<UserAgentDataInfo | null> {
  if (typeof navigator === "undefined") return null;
  const uaData = (navigator as Navigator & { userAgentData?: UADataLike })
    .userAgentData;
  if (!uaData || typeof uaData.getHighEntropyValues !== "function") return null;
  try {
    const v = await uaData.getHighEntropyValues([
      "platform",
      "platformVersion",
      "uaFullVersion",
      "model",
      "architecture",
      "bitness",
    ]);
    const s = (key: string): string | null =>
      typeof v[key] === "string" && v[key] ? (v[key] as string) : null;
    return {
      platform: s("platform"),
      platformVersion: s("platformVersion"),
      uaFullVersion: s("uaFullVersion"),
      model: s("model"),
      architecture: s("architecture"),
      bitness: s("bitness"),
    };
  } catch {
    return null;
  }
}

interface BatteryManagerLike {
  level: number;
  charging: boolean;
}

async function readBattery(): Promise<BatteryInfo | null> {
  if (typeof navigator === "undefined") return null;
  const getBattery = (
    navigator as Navigator & {
      getBattery?: () => Promise<BatteryManagerLike>;
    }
  ).getBattery;
  if (typeof getBattery !== "function") return null;
  try {
    const b = await getBattery.call(navigator);
    return {
      level: typeof b.level === "number" ? b.level : null,
      charging: typeof b.charging === "boolean" ? b.charging : null,
    };
  } catch {
    return null;
  }
}

/**
 * Sammelt den vollständigen Environment-Block. numThreads + threadingMode
 * kommen aus dem Pose-Detection-Hook (tatsächliche Laufzeitwerte), power aus
 * den manuellen UI-Toggles.
 */
export async function collectEnvironment(
  numThreads: number | null,
  threadingMode: "single" | "multi",
  power: ManualPowerInfo,
): Promise<EnvironmentInfo> {
  const [userAgentData, battery] = await Promise.all([
    readUserAgentData(),
    readBattery(),
  ]);
  const coi =
    typeof self !== "undefined" &&
    (self as unknown as { crossOriginIsolated?: boolean })
      .crossOriginIsolated === true;
  const cores =
    typeof navigator !== "undefined" ? navigator.hardwareConcurrency || 0 : 0;
  const mem =
    typeof navigator !== "undefined"
      ? (navigator as Navigator & { deviceMemory?: number }).deviceMemory ??
        null
      : null;
  return {
    capturedAt: new Date().toISOString(),
    crossOriginIsolated: coi,
    hardwareConcurrency: cores,
    deviceMemoryGb: mem,
    numThreads,
    threadingMode,
    tfjs: TFJS_VERSIONS,
    userAgentData,
    battery,
    power,
  };
}
