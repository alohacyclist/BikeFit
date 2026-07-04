/**
 * Best-Effort-Hardware-/Umgebungsinfo für den Benchmark-Export.
 *
 * Browser können weder CPU-Modell noch Gerätetyp zuverlässig auslesen — die
 * Werte sind heuristisch aus dem User-Agent + navigator abgeleitet und dienen
 * als Kontext im Ergebniskapitel. Für FF3 (Latenz) sollte der Studienleiter
 * device/cpu ggf. manuell überschreiben (siehe QuantizationControls).
 */

export interface HardwareInfo {
  device: string;
  cpu: string;
  os: string;
  browser: string;
}

function detectOs(ua: string, platform: string): string {
  if (/windows nt 10/i.test(ua)) return "Windows 10/11";
  if (/windows/i.test(ua)) return "Windows";
  if (/mac os x ([0-9_]+)/i.test(ua)) {
    const m = ua.match(/mac os x ([0-9_]+)/i);
    return "macOS " + (m ? m[1].replace(/_/g, ".") : "").trim();
  }
  if (/android ([0-9.]+)/i.test(ua)) {
    const m = ua.match(/android ([0-9.]+)/i);
    return "Android " + (m ? m[1] : "");
  }
  if (/iphone|ipad|ipod/i.test(ua)) return "iOS";
  if (/linux/i.test(ua)) return "Linux";
  return platform || "unknown";
}

function detectBrowser(ua: string): string {
  // Reihenfolge wichtig: Edge/Chrome enthalten "Safari"/"Chrome"-Tokens.
  const patterns: Array<[RegExp, string]> = [
    [/edg\/([0-9.]+)/i, "Edge"],
    [/opr\/([0-9.]+)/i, "Opera"],
    [/chrome\/([0-9.]+)/i, "Chrome"],
    [/firefox\/([0-9.]+)/i, "Firefox"],
    [/version\/([0-9.]+).*safari/i, "Safari"],
  ];
  for (const [re, name] of patterns) {
    const m = ua.match(re);
    if (m) return `${name} ${m[1]}`;
  }
  return "unknown";
}

export function detectHardware(): HardwareInfo {
  if (typeof navigator === "undefined") {
    return { device: "unknown", cpu: "unknown", os: "unknown", browser: "unknown" };
  }
  const ua = navigator.userAgent || "";
  const platform =
    (navigator as Navigator & { platform?: string }).platform || "";
  const cores = navigator.hardwareConcurrency || 0;
  const mem = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;
  const cpuParts = [
    platform || "unknown-arch",
    cores ? `${cores} cores` : null,
    mem ? `${mem} GB RAM` : null,
  ].filter(Boolean);
  return {
    device: platform || "unknown",
    cpu: cpuParts.join(", "),
    os: detectOs(ua, platform),
    browser: detectBrowser(ua),
  };
}

/**
 * ISO-8601-Timestamp → kompaktes Dateinamens-Format YYYYMMDDTHHMMSSZ
 * (keine ':' / '.', UTC). Beispiel: 2026-07-04T12:34:56.789Z → 20260704T123456Z.
 */
export function toCompactTimestamp(iso: string): string {
  return iso.replace(/[-:]/g, "").replace(/\.\d+Z$/, "Z");
}
