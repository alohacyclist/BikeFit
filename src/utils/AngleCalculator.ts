/**
 * AngleCalculator — Vektor-basierte Berechnung des Kniewinkels.
 *
 * Scope-Reduktion: für die Bachelorarbeit wird ausschließlich der
 * Kniewinkel untersucht. Frühere Winkel-Funktionen (Hüfte, Knöchel,
 * Ellbogen, Rücken) wurden bewusst entfernt, um die Mess-App schlank
 * und auf eine abhängige Variable fokussiert zu halten.
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
  p3: Point2D
): number {
  const vectorA = { x: p1.x - vertex.x, y: p1.y - vertex.y };
  const vectorB = { x: p3.x - vertex.x, y: p3.y - vertex.y };
  const dotProduct = vectorA.x * vectorB.x + vectorA.y * vectorB.y;
  const magnitudeA = Math.sqrt(
    vectorA.x * vectorA.x + vectorA.y * vectorA.y
  );
  const magnitudeB = Math.sqrt(
    vectorB.x * vectorB.x + vectorB.y * vectorB.y
  );
  if (magnitudeA === 0 || magnitudeB === 0) return 0;
  let cosAngle = dotProduct / (magnitudeA * magnitudeB);
  cosAngle = Math.max(-1, Math.min(1, cosAngle));
  return Math.acos(cosAngle) * (180 / Math.PI);
}

/**
 * Berechnet den Kniewinkel zwischen Hüfte, Knie und Knöchel.
 * Liefert null wenn einer der drei Keypoint-Scores unter minConfidence liegt.
 */
export function calculateKneeAngle(
  hip: Keypoint,
  knee: Keypoint,
  ankle: Keypoint,
  minConfidence: number = 0.2
): number | null {
  if (
    (hip.score !== undefined && hip.score < minConfidence) ||
    (knee.score !== undefined && knee.score < minConfidence) ||
    (ankle.score !== undefined && ankle.score < minConfidence)
  ) {
    return null;
  }
  return calculateAngle(hip, knee, ankle);
}

/**
 * Reduzierte Winkel-Datenstruktur — nur Knie für diese Studie.
 */
export interface BiomechanicalAngles {
  knee: number | null;
}

export interface AngleRange {
  min: number;
  max: number;
  label: string;
}

/** Optimaler Bereich am unteren Totpunkt (Knie maximal gestreckt). */
export const KNEE_EXTENSION_RANGE: AngleRange = {
  min: 140,
  max: 150,
  label: 'Kniewinkel (gestreckt)',
};

/** Typischer Bereich am oberen Totpunkt (Knie maximal gebeugt). */
export const KNEE_FLEXION_RANGE: AngleRange = {
  min: 65,
  max: 75,
  label: 'Kniewinkel (gebeugt)',
};

export type AngleStatus = 'optimal' | 'acceptable' | 'critical';

export function evaluateAngle(
  angle: number,
  range: AngleRange,
  tolerance: number = 5
): AngleStatus {
  if (angle >= range.min && angle <= range.max) return 'optimal';
  if (
    angle >= range.min - tolerance &&
    angle <= range.max + tolerance
  )
    return 'acceptable';
  return 'critical';
}

export function getStatusColor(status: AngleStatus): string {
  switch (status) {
    case 'optimal':
      return '#22c55e';
    case 'acceptable':
      return '#eab308';
    case 'critical':
      return '#ef4444';
  }
}

export function getAngleColor(angle: number): string {
  if (angle >= 140 && angle <= 150) return '#22c55e';
  if (angle >= 135 && angle <= 155) return '#eab308';
  return '#ef4444';
}

/** Körperseite, die getrackt wird. */
export type BodySide = 'left' | 'right';

export const SIDE_KEYPOINTS: Record<
  BodySide,
  { hip: number; knee: number; ankle: number }
> = {
  left: { hip: 11, knee: 13, ankle: 15 },
  right: { hip: 12, knee: 14, ankle: 16 },
};

/**
 * 1 wenn hip+knee+ankle alle über minConfidence liegen, sonst 0.
 * Über Setup-Fenster gemittelt = prospektive Validity-Rate.
 */
export function isKneeTripleValid(
  keypoints: Keypoint[],
  side: BodySide,
  minConfidence = 0.2
): 0 | 1 {
  const ix = SIDE_KEYPOINTS[side];
  const hip = keypoints[ix.hip]?.score ?? 0;
  const knee = keypoints[ix.knee]?.score ?? 0;
  const ankle = keypoints[ix.ankle]?.score ?? 0;
  return hip >= minConfidence &&
    knee >= minConfidence &&
    ankle >= minConfidence
    ? 1
    : 0;
}
