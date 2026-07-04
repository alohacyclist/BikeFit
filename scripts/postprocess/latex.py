"""LaTeX-Tabellen via pandas.to_latex(), Dezimalkomma per Post-hoc-Replace.

Das Dezimalkomma wird NACH to_latex() mit Regex (\\d)\\.(\\d) → \\1{,}\\2
gesetzt. Da die Tabellen ausschließlich Zahlen + Textlabels enthalten und
Labels keine 'Ziffer.Ziffer'-Muster tragen, ist der Replace sicher.
"""

from __future__ import annotations

import re

import pandas as pd

_DECIMAL_RE = re.compile(r"(\d)\.(\d)")


def to_latex_decimal_comma(
    df: pd.DataFrame,
    caption: str,
    label: str,
    float_format: str = "%.2f",
) -> str:
    """DataFrame → LaTeX-Tabelle mit deutschem Dezimalkomma.

    Bewusst OHNE to_latex(caption=..., label=...): diese Argumente erzwingen in
    pandas den Styler-Pfad (jinja2-Abhängigkeit). Stattdessen wird das schlichte
    tabular manuell in ein table-Environment mit Caption/Label eingebettet.
    """
    tabular = df.to_latex(
        index=False,
        escape=True,
        float_format=lambda value: float_format % value,
        na_rep="–",
    )
    tabular = _DECIMAL_RE.sub(r"\1{,}\2", tabular)
    return (
        "\\begin{table}[ht]\n\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"{tabular}"
        "\\end{table}\n"
    )
