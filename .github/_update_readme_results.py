"""Befüllt die README-Ergebnistabellen (Abschnitt 6.1 & 6.3) automatisch aus
den CSV-Dateien einer Notebook-Ausführung.

Voraussetzung: Das Notebook wurde bereits ausgeführt (siehe README, Abschnitt 8),
sodass `results/metrics_comparison.csv` und `results/stationarity_tests.csv`
existieren.

Nutzung:
    jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=1200 NVIDIA_Kursprognose_LSTM.ipynb
    python .github/_update_readme_results.py

Idempotent: kann beliebig oft erneut ausgeführt werden, ersetzt jeweils nur
den Inhalt zwischen den `<!-- AUTO-GENERATED:...:START/END -->`-Markern.
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
README_PATH = ROOT / "README.md"


def results_dir():
    for candidate in (ROOT / "notebooks" / "results", ROOT / "results"):
        if candidate.is_dir():
            return candidate
    return None


def replace_block(text, marker, new_body):
    start = f"<!-- AUTO-GENERATED:{marker}:START -->"
    end = f"<!-- AUTO-GENERATED:{marker}:END -->"
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    if not pattern.search(text):
        raise SystemExit(f"Marker '{marker}' nicht in README.md gefunden.")
    return pattern.sub(f"{start}\n{new_body}\n{end}", text)


# ── 6.1  Modellvergleich ─────────────────────────────────────────────────────

MODEL_LABELS = [
    ("Naiv",               "Naive Baseline"),
    ("Moving Average",     "Moving Average"),
    ("ARIMA",              "ARIMA/SARIMA"),
    ("LSTM (Log-Rendite",  "LSTM (Log-Renditen)"),
]
MODEL_ROW_ORDER = ["Naive Baseline", "Moving Average", "ARIMA/SARIMA",
                    "LSTM (Niveau)", "LSTM (Log-Renditen)"]


def label_for(model_key):
    for prefix, label in MODEL_LABELS:
        if model_key.startswith(prefix):
            return label
    return "LSTM (Niveau)"   # dynamischer HPO-Name, z.B. "LSTM* (64,)/d0.2"


def render_metrics_table(csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    rows = {label_for(idx): row for idx, row in df.iterrows()}

    col_w = (20, 9, 10, 8, 9, 6)
    header = (f"{'Modell':<{col_w[0]}} │ {'MAE (USD)':>{col_w[1]}} │ "
              f"{'RMSE (USD)':>{col_w[2]}} │ {'MAPE (%)':>{col_w[3]}} │ "
              f"{'Theil´s U':>{col_w[4]}} │ {'DA (%)':>{col_w[5]}}")
    sep = ("─" * (col_w[0] + 1) + "┼" + "─" * (col_w[1] + 2) + "┼" +
           "─" * (col_w[2] + 2) + "┼" + "─" * (col_w[3] + 2) + "┼" +
           "─" * (col_w[4] + 2) + "┼" + "─" * (col_w[5] + 2))

    lines = [header, sep]
    for label in MODEL_ROW_ORDER:
        r = rows.get(label)
        if r is None:
            lines.append(f"{label:<{col_w[0]}} │ {'–':>{col_w[1]}} │ {'–':>{col_w[2]}} │ "
                         f"{'–':>{col_w[3]}} │ {'–':>{col_w[4]}} │ {'–':>{col_w[5]}}")
            continue
        mae, rmse, mape, theil = r["MAE"], r["RMSE"], r["MAPE"] * 100, r["Theil_U"]
        lines.append(f"{label:<{col_w[0]}} │ {mae:{col_w[1]}.3f} │ {rmse:{col_w[2]}.3f} │ "
                     f"{mape:{col_w[3]}.2f} │ {theil:{col_w[4]}.4f} │ {'–':>{col_w[5]}}")
    lines.append(sep.replace("┼", "┴"))
    lines.append("Automatisch befüllt aus results/metrics_comparison.csv "
                 f"({pd.Timestamp.now():%Y-%m-%d})")
    return "```\n" + "\n".join(lines) + "\n```"


# ── 6.3  Stationaritätsanalyse ───────────────────────────────────────────────

STATIONARITY_ROW_ORDER = ["Close (Niveau)", "Log-Renditen", "Einfache Renditen"]


def fmt_p(p):
    if p < 0.001:
        return "< 0.001"
    if p < 0.01:
        return "< 0.01"
    return f"{p:.3g}"


def kpss_befund(kpss_p):
    return "stationär" if kpss_p >= 0.05 else "stationär abgelehnt"


def schluss(verdict):
    v = verdict.strip()
    if v.startswith("NICHT STATIONÄR"):
        return "**nicht stationär**"
    if v.startswith("STATIONÄR"):
        return "**stationär**"
    if v.startswith("TRENDDSTATIONÄR"):
        return "trendstationär?"
    return "unschlüssig"


def render_stationarity_table(csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    lines = ["| Zeitreihe | ADF p-Wert | KPSS-Befund | Schluss |",
             "|-----------|-----------|-------------|---------|"]
    for name in STATIONARITY_ROW_ORDER:
        if name not in df.index:
            lines.append(f"| {name} | – | – | – |")
            continue
        row = df.loc[name]
        lines.append(f"| {name} | {fmt_p(row['ADF_p'])} | "
                     f"{kpss_befund(row['KPSS_p'])} | {schluss(row['Verdict'])} |")
    return "\n".join(lines)


def main():
    rdir = results_dir()
    if rdir is None:
        print("Kein results/-Verzeichnis gefunden. Notebook zuerst ausführen "
              "(siehe README Abschnitt 8).", file=sys.stderr)
        return 1

    text = README_PATH.read_text(encoding="utf-8")
    updated = False

    metrics_csv = rdir / "metrics_comparison.csv"
    if metrics_csv.exists():
        text = replace_block(text, "METRICS_TABLE", render_metrics_table(metrics_csv))
        print(f"Metriktabelle aus {metrics_csv} übernommen.")
        updated = True
    else:
        print(f"{metrics_csv} nicht gefunden -> Metriktabelle übersprungen.")

    stationarity_csv = rdir / "stationarity_tests.csv"
    if stationarity_csv.exists():
        text = replace_block(text, "STATIONARITY_TABLE",
                             render_stationarity_table(stationarity_csv))
        print(f"Stationaritätstabelle aus {stationarity_csv} übernommen.")
        updated = True
    else:
        print(f"{stationarity_csv} nicht gefunden -> Stationaritätstabelle übersprungen.")

    if updated:
        README_PATH.write_text(text, encoding="utf-8")
        print("README.md aktualisiert.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
