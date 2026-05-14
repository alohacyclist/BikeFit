/**
 * AngleCalculator - Biometrische Winkelberechnung mit Vektorgeometrie
 *
 * Nutzt das Skalarprodukt zur robusten Winkelberechnung,
 * unabhängig von der Kameraneigung.
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
 * Berechnet den Winkel zwischen drei Punkten (in Grad).
 * Der Winkel wird am mittleren Punkt (vertex) gemessen.
 *
 * Formel: θ = acos((A · B) / (|A| × |B|))
 * wobei A = p1 - vertex und B = p3 - vertex
 *
 * @param p1 - Erster Punkt (z.B. hip)
 * @param vertex - Scheitelpunkt/Gelenk (z.B. knee)
 * @param p3 - Dritter Punkt (z.B. ankle)
 * @returns Winkel in Grad (0-180)
 */
export function calculateAngle(p1: Point2D, vertex: Point2D, p3: Point2D): number {
  // Vektor A: von vertex zu p1
  const vectorA = {
    x: p1.x - vertex.x,
    y: p1.y - vertex.y
  };

  // Vektor B: von vertex zu p3
  const vectorB = {
    x: p3.x - vertex.x,
    y: p3.y - vertex.y
  };

  // Skalarprodukt: A · B = Ax*Bx + Ay*By
  const dotProduct = vectorA.x * vectorB.x + vectorA.y * vectorB.y;

  // Beträge (Längen) der Vektoren
  const magnitudeA = Math.sqrt(vectorA.x * vectorA.x + vectorA.y * vectorA.y);
  const magnitudeB = Math.sqrt(vectorB.x * vectorB.x + vectorB.y * vectorB.y);

  // Division durch Null vermeiden
  if (magnitudeA === 0 || magnitudeB === 0) {
    return 0;
  }

  // cos(θ) = (A · B) / (|A| × |B|)
  let cosAngle = dotProduct / (magnitudeA * magnitudeB);

  // Numerische Stabilität: Wert auf [-1, 1] begrenzen
  cosAngle = Math.max(-1, Math.min(1, cosAngle));

  // Winkel in Grad umrechnen
  const angleRadians = Math.acos(cosAngle);
  const angleDegrees = angleRadians * (180 / Math.PI);

  return angleDegrees;
}

/**
 * Berechnet den Kniewinkel aus MoveNet Keypoints.
 * Keypoint-Indizes für MoveNet:
 * - hip: 11 (left) oder 12 (right)
 * - knee: 13 (left) oder 14 (right)
 * - ankle: 15 (left) oder 16 (right)
 */
export function calculateKneeAngle(
  hip: Keypoint,
  knee: Keypoint,
  ankle: Keypoint,
  minConfidence: number = 0.3
): number | null {
  // Confidence-Check
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
 * Berechnet den Hüftwinkel (Oberkörperneigung).
 * Keypoint-Indizes für MoveNet:
 * - shoulder: 5 (left) oder 6 (right)
 * - hip: 11 (left) oder 12 (right)
 * - knee: 13 (left) oder 14 (right)
 */
export function calculateHipAngle(
  shoulder: Keypoint,
  hip: Keypoint,
  knee: Keypoint,
  minConfidence: number = 0.3
): number | null {
  if (
    (shoulder.score !== undefined && shoulder.score < minConfidence) ||
    (hip.score !== undefined && hip.score < minConfidence) ||
    (knee.score !== undefined && knee.score < minConfidence)
  ) {
    return null;
  }

  return calculateAngle(shoulder, hip, knee);
}

/**
 * Berechnet den Knöchelwinkel (Fußstellung).
 * Keypoint-Indizes für MoveNet:
 * - knee: 13 (left) oder 14 (right)
 * - ankle: 15 (left) oder 16 (right)
 * - foot: simuliert durch Verlängerung vom Knöchel nach unten
 */
export function calculateAnkleAngle(
  knee: Keypoint,
  ankle: Keypoint,
  minConfidence: number = 0.3
): number | null {
  if (
    (knee.score !== undefined && knee.score < minConfidence) ||
    (ankle.score !== undefined && ankle.score < minConfidence)
  ) {
    return null;
  }

  // Virtueller Fußpunkt (horizontal vom Knöchel)
  const footPoint: Point2D = {
    x: ankle.x + 50,
    y: ankle.y
  };

  return calculateAngle(knee, ankle, footPoint);
}

/**
 * Berechnet den Ellenbogenwinkel (Armhaltung).
 * Keypoint-Indizes für MoveNet:
 * - shoulder: 5 (left) oder 6 (right)
 * - elbow: 7 (left) oder 8 (right)
 * - wrist: 9 (left) oder 10 (right)
 */
export function calculateElbowAngle(
  shoulder: Keypoint,
  elbow: Keypoint,
  wrist: Keypoint,
  minConfidence: number = 0.3
): number | null {
  if (
    (shoulder.score !== undefined && shoulder.score < minConfidence) ||
    (elbow.score !== undefined && elbow.score < minConfidence) ||
    (wrist.score !== undefined && wrist.score < minConfidence)
  ) {
    return null;
  }

  return calculateAngle(shoulder, elbow, wrist);
}

/**
 * Berechnet den Rückenwinkel (Torso zur Horizontalen).
 * Misst den Winkel zwischen Schulter-Hüfte-Linie und der Horizontalen.
 */
export function calculateBackAngle(
  shoulder: Keypoint,
  hip: Keypoint,
  minConfidence: number = 0.3
): number | null {
  if (
    (shoulder.score !== undefined && shoulder.score < minConfidence) ||
    (hip.score !== undefined && hip.score < minConfidence)
  ) {
    return null;
  }

  // Vektor von Hüfte zu Schulter
  const dx = shoulder.x - hip.x;
  const dy = shoulder.y - hip.y;

  // Winkel zur Vertikalen (Y-Achse zeigt nach unten im Canvas)
  const angleToVertical = Math.atan2(dx, -dy) * (180 / Math.PI);

  // Winkel zur Horizontalen = 90° - Winkel zur Vertikalen
  const angleToHorizontal = 90 - Math.abs(angleToVertical);

  return Math.abs(angleToHorizontal);
}

/**
 * Interface für alle biomechanischen Winkel
 */
export interface BiomechanicalAngles {
  knee: number | null;
  hip: number | null;
  ankle: number | null;
  elbow: number | null;
  back: number | null;
}

/**
 * Optimale Winkelbereiche für Bike-Fitting
 */
export interface AngleRange {
  min: number;
  max: number;
  label: string;
}

export const OPTIMAL_RANGES: Record<keyof BiomechanicalAngles, AngleRange> = {
  knee: { min: 140, max: 150, label: 'Kniewinkel (gestreckt)' },
  hip: { min: 40, max: 50, label: 'Hüftwinkel' },
  ankle: { min: 90, max: 110, label: 'Knöchelwinkel' },
  elbow: { min: 150, max: 170, label: 'Ellenbogenwinkel' },
  back: { min: 40, max: 50, label: 'Rückenneigung' }
};

/**
 * Berechnet den Knie-Beugungswinkel (am oberen Totpunkt)
 * Typischer Bereich: 65° - 75°
 */
export const KNEE_FLEXION_RANGE: AngleRange = {
  min: 65,
  max: 75,
  label: 'Kniewinkel (gebeugt)'
};

/**
 * Prüft ob der Kniewinkel im optimalen Bereich für Bike-Fitting liegt.
 * Typischer Bereich: 140° - 150° bei gestrecktem Bein (unterer Totpunkt)
 */
export function isKneeAngleOptimal(angle: number): boolean {
  return angle >= 140 && angle <= 150;
}

/**
 * Bewertungsstufe für einen Winkel
 */
export type AngleStatus = 'optimal' | 'acceptable' | 'critical';

/**
 * Bewertet einen Winkel basierend auf dem optimalen Bereich
 */
export function evaluateAngle(
  angle: number,
  range: AngleRange,
  tolerance: number = 5
): AngleStatus {
  if (angle >= range.min && angle <= range.max) {
    return 'optimal';
  } else if (angle >= range.min - tolerance && angle <= range.max + tolerance) {
    return 'acceptable';
  }
  return 'critical';
}

/**
 * Gibt eine Farbe basierend auf dem Winkelstatus zurück.
 */
export function getStatusColor(status: AngleStatus): string {
  switch (status) {
    case 'optimal':
      return '#22c55e'; // green-500
    case 'acceptable':
      return '#eab308'; // yellow-500
    case 'critical':
      return '#ef4444'; // red-500
  }
}

/**
 * Gibt eine Farbe basierend auf dem Kniewinkel zurück.
 * Grün: optimal (140-150°)
 * Gelb: akzeptabel (135-155°)
 * Rot: außerhalb
 */
export function getAngleColor(angle: number): string {
  if (angle >= 140 && angle <= 150) {
    return '#22c55e'; // green-500
  } else if (angle >= 135 && angle <= 155) {
    return '#eab308'; // yellow-500
  }
  return '#ef4444'; // red-500
}
