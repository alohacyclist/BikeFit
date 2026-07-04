"""Post-Processing für BikeFit-Benchmark-Exporte (Vertrag v1.0.0).

Aggregiert die Roh-JSON-Exporte der Messpipeline gegen eine Tracker-Ground-
Truth und berechnet die Kennzahlen für FF1 (Baseline-Genauigkeit), FF2
(Quantisierungseffekt) und FF3 (Inferenzlatenz).

Design: N=1, deskriptiv-explorativ. Keine Inferenzstatistik (kein Friedman,
keine Wilcoxon-Post-hocs, keine Effektgrößen) — bewusste Entscheidung wegen
N=1 + Autokorrelation. Perzentile manuell (Nearest-Rank), scipy nur für die
BDC-Peak-Detektion.
"""

from .schema import SCHEMA_VERSION

__all__ = ["SCHEMA_VERSION"]
