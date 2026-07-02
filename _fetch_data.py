"""Lädt den in _build_notebook.py (Zelle 0.1) konfigurierten Ticker/Zeitraum
via yfinance und speichert ihn als CSV-Snapshot unter data/nvda_ohlcv.csv.

Dient als Offline-Fallback für load_price_data() im Notebook, falls kein
Live-Netzwerkzugriff auf Yahoo Finance besteht (siehe README, Abschnitt 8).

Nutzung:
    python _fetch_data.py
"""
import re
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent
BUILD_SCRIPT = ROOT / "_build_notebook.py"
DATA_DIR = ROOT / "data"
OUT_PATH = DATA_DIR / "nvda_ohlcv.csv"


def read_config():
    text = BUILD_SCRIPT.read_text(encoding="utf-8")
    ticker = re.search(r'^TICKER\s*=\s*"([^"]+)"', text, re.M).group(1)
    start = re.search(r'^START_DATE\s*=\s*"([^"]+)"', text, re.M).group(1)
    end = re.search(r'^END_DATE\s*=\s*"([^"]+)"', text, re.M).group(1)
    return ticker, start, end


def main():
    ticker, start, end = read_config()
    print(f"Lade {ticker} von {start} bis {end} via yfinance ...")

    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if raw is None or len(raw) == 0:
        print("Kein Netzwerkzugriff auf Yahoo Finance oder keine Daten erhalten.",
              file=sys.stderr)
        return 1

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "Date"

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH)
    print(f"Gespeichert: {OUT_PATH}  ({len(df)} Handelstage, "
          f"{df.index.min().date()} bis {df.index.max().date()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
