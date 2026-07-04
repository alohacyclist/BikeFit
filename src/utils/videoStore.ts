/**
 * IndexedDB-Speicher für (a) die aktuelle Replay-Datei und (b) die während
 * einer Voll-Sequenz erzeugten Ergebnis-JSONs.
 *
 * Der Stufenwechsel läuft über window.location.reload() (frischer
 * tfjs-tflite-Pthread-Pool → kein Delegate-Swap-Deadlock). Ein Reload verwirft
 * das in-Memory File-Objekt UND macht einen automatischen Download un-gestured
 * (Chrome blockt/verwirft ihn). Daher:
 *   - Video über Reloads hinweg hier zwischenspeichern.
 *   - Ergebnis-JSONs sammeln und am Sequenzende per EINER User-Geste
 *     (Button-Klick) herunterladen — nicht automatisch nach jedem Reload.
 */

const DB_NAME = "edgefit";
const DB_VERSION = 2;
const VIDEO_STORE = "video";
const RESULT_STORE = "results";
const VIDEO_KEY = "current";

export interface StoredResult {
  level: string;
  filename: string;
  json: string;
}

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(VIDEO_STORE)) {
        db.createObjectStore(VIDEO_STORE);
      }
      if (!db.objectStoreNames.contains(RESULT_STORE)) {
        db.createObjectStore(RESULT_STORE, { keyPath: "level" });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function tx<T>(
  store: string,
  mode: IDBTransactionMode,
  run: (s: IDBObjectStore) => IDBRequest<T> | null,
): Promise<T | void> {
  return openDb().then(
    (db) =>
      new Promise<T | void>((resolve, reject) => {
        const transaction = db.transaction(store, mode);
        const request = run(transaction.objectStore(store));
        transaction.oncomplete = () =>
          resolve(request ? request.result : undefined);
        transaction.onerror = () => reject(transaction.error);
        transaction.onabort = () => reject(transaction.error);
      }).finally(() => db.close()),
  );
}

// --- Video ---
export async function saveVideo(file: File): Promise<void> {
  await tx(VIDEO_STORE, "readwrite", (s) => s.put(file, VIDEO_KEY));
}

export async function loadVideo(): Promise<File | null> {
  const value = await tx<File>(VIDEO_STORE, "readonly", (s) =>
    s.get(VIDEO_KEY),
  );
  return value instanceof File ? value : null;
}

export async function clearVideo(): Promise<void> {
  await tx(VIDEO_STORE, "readwrite", (s) => s.delete(VIDEO_KEY));
}

// --- Ergebnis-JSONs (Voll-Sequenz) ---
export async function saveResult(
  level: string,
  filename: string,
  json: string,
): Promise<void> {
  await tx(RESULT_STORE, "readwrite", (s) =>
    s.put({ level, filename, json } as StoredResult),
  );
}

export async function loadResults(): Promise<StoredResult[]> {
  const all = await tx<StoredResult[]>(RESULT_STORE, "readonly", (s) =>
    s.getAll(),
  );
  return Array.isArray(all) ? all : [];
}

export async function clearResults(): Promise<void> {
  await tx(RESULT_STORE, "readwrite", (s) => s.clear());
}
