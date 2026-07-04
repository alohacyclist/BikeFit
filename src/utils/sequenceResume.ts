/**
 * Reload-übergreifender Zustand für den Stufenwechsel / die Voll-Sequenz.
 *
 * Da jeder Stufenwechsel per window.location.reload() einen frischen
 * tfjs-tflite-Pthread-Pool erzwingt (Fix des qu8-Delegate-Swap-Deadlocks),
 * muss die "beim nächsten Boot zu ladende Stufe" und der Rest der Sequenz
 * einen Reload überleben. sessionStorage ist tab-lokal und dafür ideal.
 */

import type { QuantizationLevel } from "../types/quantization";

const KEY = "edgefit.pendingBoot";
const LEVELS: QuantizationLevel[] = ["fp32", "fp16", "int8"];

export interface PendingBoot {
  /** Stufe, die dieser Boot laden soll. */
  level: QuantizationLevel;
  /** Noch ausstehende Stufen NACH dieser (leere Liste = letzte / Einzelwechsel). */
  remaining: QuantizationLevel[];
  /** true = Voll-Sequenz: nach Boot automatisch aufnehmen + fortschalten. */
  autoRecord: boolean;
}

function isLevel(value: unknown): value is QuantizationLevel {
  return typeof value === "string" && (LEVELS as string[]).includes(value);
}

export function readPending(): PendingBoot | null {
  try {
    const raw = sessionStorage.getItem(KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (
      !isLevel(parsed?.level) ||
      !Array.isArray(parsed?.remaining) ||
      !parsed.remaining.every(isLevel) ||
      typeof parsed?.autoRecord !== "boolean"
    ) {
      return null;
    }
    return {
      level: parsed.level,
      remaining: parsed.remaining,
      autoRecord: parsed.autoRecord,
    };
  } catch {
    return null;
  }
}

export function writePending(pending: PendingBoot): void {
  try {
    sessionStorage.setItem(KEY, JSON.stringify(pending));
  } catch {
    // sessionStorage nicht verfügbar — dann kein Resume möglich, aber der
    // Reload lädt zumindest die Default-Stufe (fp32).
  }
}

export function clearPending(): void {
  try {
    sessionStorage.removeItem(KEY);
  } catch {
    // ignore
  }
}
