/**
 * SHA-256-Fingerprint einer Datei via WebCrypto.
 * Verwendet zur Modell-Identifikation im Benchmark-Export.
 */

export interface ModelFingerprint {
  url: string;
  sha256: string;
  sizeBytes: number;
  loadedAt: string;
}

function bytesToHex(bytes: ArrayBuffer): string {
  const arr = new Uint8Array(bytes);
  let hex = '';
  for (let i = 0; i < arr.length; i++) {
    hex += arr[i].toString(16).padStart(2, '0');
  }
  return hex;
}

export async function fingerprintModel(url: string): Promise<ModelFingerprint> {
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(`Modell ${url} nicht erreichbar (HTTP ${resp.status})`);
  }
  const buf = await resp.arrayBuffer();
  const digest = await crypto.subtle.digest('SHA-256', buf);
  return {
    url,
    sha256: bytesToHex(digest),
    sizeBytes: buf.byteLength,
    loadedAt: new Date().toISOString(),
  };
}
