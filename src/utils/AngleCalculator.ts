/**
 * AngleCalculator — Vektor-basierte Berechnung des Kniewinkels.
 *
 * Reiner Datensammler: liefert für jedes Tripel (hip, knee, ankle) einen
 * Winkel zurück, OHNE Confidence-Filter. Selektion (z.B. Frames mit
 * unzuverlässigen Keypoint-Scores ausschließen) erfolgt erst im
 * Python-Postprocessing — dort als dokumentierter, variierbarer Parameter.
 *
 * Begründung: Ein app-seitiger Filter VOR der Winkelberechnung würde
 * modellabhängige Teilmengen erzeugen (INT8 verliert systematisch mehr
 * Frames als FP32) und damit den paired-Vergleich der Quantisierungsstufen
 * verfälschen (Selection-Bias).
 */

export interface Point2D {
  x: number;
  y: number;
}

export interface Keypoint extends Point2D {
  score?: number;
  name?: string;
}

/**
 * Innerer Winkel zwischen drei Punkten in Grad (am vertex gemessen).
 * Formel: θ = acos((A·B) / (|A|·|B|))
 */
export function calculateAngle(
  p1: Point2D,
  vertex: Point2D,
  p3: Point2D,
): number {
  const vectorA = { x: p1.x - vertex.x, y: p1.y - vertex.y };
  const vectorB = { x: p3.x - vertex.x, y: p3.y - vertex.y };
  const dotProduct = vectorA.x * vectorB.x + vectorA.y * vectorB.y;
  const magnitudeA = Math.sqrt(vectorA.x * vectorA.x + vectorA.y * vectorA.y);
  const magnitudeB = Math.sqrt(vectorB.x * vectorB.x + vectorB.y * vectorB.y);
  if (magnitudeA === 0 || magnitudeB === 0) return 0;
  let cosAngle = dotProduct / (magnitudeA * magnitudeB);
  cosAngle = Math.max(-1, Math.min(1, cosAngle));
  return Math.acos(cosAngle) * (180 / Math.PI);
}

/**
 * Kniewinkel zwischen Hüfte, Knie und Knöchel — immer berechnet, kein Filter.
 * Postprocessing entscheidet anhand der Keypoint-Scores (im Export pro Frame
 * enthalten), welche Werte für die Auswertung gelten.
 */
export function calculateKneeAngle(
  hip: Keypoint,
  knee: Keypoint,
  ankle: Keypoint,
): number {
  return calculateAngle(hip, knee, ankle);
}

/** Körperseite, die getrackt wird. */
export type BodySide = "left" | "right";

export const SIDE_KEYPOINTS: Record<
  BodySide,
  { hip: number; knee: number; ankle: number }
> = {
  left: { hip: 11, knee: 13, ankle: 15 },
  right: { hip: 12, knee: 14, ankle: 16 },
};
