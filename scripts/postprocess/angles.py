"""Kniewinkel — Vektorformel, IDENTISCH zur Pipeline (src/utils/AngleCalculator.ts).

θ = arccos((BA·BC) / (|BA|·|BC|)) · 180/π, Scheitel B = Knie.
Innenwinkel 0..180°. Y-Achsen-Konvention (oben/unten) ist irrelevant: der
Innenwinkel ist reflexionsinvariant. Dieselbe Funktion wird für die
GT-Berechnung genutzt (Konsistenzforderung Pipeline == GT).
"""

from __future__ import annotations

import math
from typing import Tuple

Point = Tuple[float, float]


def calculate_angle(p1: Point, vertex: Point, p3: Point) -> float:
    """Innenwinkel am vertex zwischen p1 und p3, in Grad."""
    ax, ay = p1[0] - vertex[0], p1[1] - vertex[1]
    bx, by = p3[0] - vertex[0], p3[1] - vertex[1]
    mag_a = math.hypot(ax, ay)
    mag_b = math.hypot(bx, by)
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    cos_angle = (ax * bx + ay * by) / (mag_a * mag_b)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def knee_angle(hip: Point, knee: Point, ankle: Point) -> float:
    """Kniewinkel Hüfte–Knie–Sprunggelenk (Scheitel = Knie)."""
    return calculate_angle(hip, knee, ankle)


# COCO-Keypoint-Indizes je Körperseite (identisch SIDE_KEYPOINTS in TS).
SIDE_KEYPOINTS = {
    "left": {"hip": 11, "knee": 13, "ankle": 15},
    "right": {"hip": 12, "knee": 14, "ankle": 16},
}

# Pipeline-Winkelspalte je Seite (Feldname im FrameMeasurement-Export).
SIDE_ANGLE_COLUMN = {
    "left": "kneeAngleLeft",
    "right": "kneeAngleRight",
}
