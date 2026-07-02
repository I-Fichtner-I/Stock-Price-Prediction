"""Gemeinsame Hilfsfunktionen für die Streamlit-App.

Data-Loading folgt derselben 3-Stufen-Fallback-Logik wie im Notebook
(NVIDIA_Kursprognose_LSTM.ipynb, Kap. 3.1): Live-yfinance -> CSV-Snapshot
-> synthetischer Ersatzdatensatz. Damit läuft die App auch ohne
Netzwerkzugriff.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"

TICKER = "NVDA"
SEED = 42


def _synthetic_nvda(start: str, end: str, seed: int = SEED) -> pd.DataFrame:
    """Synthetischer OHLCV-Ersatzdatensatz (nur Fallback ohne Netzwerk/Snapshot).

    Identische Methodik wie im Notebook (geometrische Brownsche Bewegung
    mit Trend-/Volatilitätsregime), damit die App auch offline konsistente
    Ergebnisse liefert.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)
    t = np.linspace(0, 1, n)
    drift = 0.0004 + 0.0010 * t**2
    vol = 0.018 + 0.015 * (t > 0.6)
    shocks = rng.normal(0, 1, n)
    log_ret = drift + vol * shocks
    close = 5.0 * np.exp(np.cumsum(log_ret))
    daily_range = np.abs(rng.normal(0, 1, n)) * vol * close
    open_ = close * (1 + rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) + daily_range * 0.5
    low = np.minimum(open_, close) - daily_range * 0.5
    volume = (rng.lognormal(mean=18.0, sigma=0.4, size=n)
              * (1 + 3 * (np.abs(log_ret) > 0.05))).astype(np.int64)
    df = pd.DataFrame({"Open": open_, "High": high, "Low": low,
                        "Close": close, "Volume": volume}, index=dates)
    df.index.name = "Date"
    return df


def load_live_price_data(ticker: str = TICKER, start: str = "2015-01-01",
                          end: str | None = None) -> tuple[pd.DataFrame, str]:
    """Lädt aktuelle OHLCV-Daten: Live-yfinance -> CSV-Snapshot -> synthetisch.

    Rückgabe: (df, source) mit source in {"yfinance", "csv_snapshot", "synthetic"}.
    """
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    df, source = None, "synthetic"
    try:
        import yfinance as yf
        raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if raw is not None and len(raw) > 0:
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.index.name = "Date"
            source = "yfinance"
    except Exception:
        df = None

    if (df is None or len(df) == 0) and ticker.upper() == TICKER:
        # CSV-Snapshot ist nur fuer NVDA vorhanden - bei anderen Tickern wuerde
        # er sonst stillschweigend als deren Daten ausgegeben werden.
        snapshot_path = DATA_DIR / "nvda_ohlcv.csv"
        if snapshot_path.is_file():
            df = pd.read_csv(snapshot_path, index_col="Date", parse_dates=True)
            df = df.loc[df.index >= start]
            if len(df) > 0:
                source = "csv_snapshot"
            else:
                df = None

    if df is None or len(df) == 0:
        df = _synthetic_nvda(start, end)
        source = "synthetic"

    return df, source


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ergänzt dieselben technischen Indikatoren wie im Notebook (Kap. 5.1)."""
    feat = df.copy()
    feat["LogReturn"] = np.log(feat["Close"]).diff()
    feat["MA20"] = feat["Close"].rolling(20).mean()
    feat["MA50"] = feat["Close"].rolling(50).mean()
    feat["Volatility20"] = feat["LogReturn"].rolling(20).std()

    delta = feat["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    feat["RSI14"] = 100 - 100 / (1 + gain / (loss + 1e-8))

    ema12 = feat["Close"].ewm(span=12, adjust=False).mean()
    ema26 = feat["Close"].ewm(span=26, adjust=False).mean()
    feat["MACD"] = ema12 - ema26

    bb_std = feat["Close"].rolling(20).std()
    bb_upper = feat["MA20"] + 2 * bb_std
    bb_lower = feat["MA20"] - 2 * bb_std
    feat["BB_Pct"] = (feat["Close"] - bb_lower) / (bb_upper - bb_lower + 1e-8)
    return feat


def load_results_csv(name: str) -> pd.DataFrame | None:
    """Lädt eine der vom Notebook exportierten results/*.csv-Dateien, falls vorhanden.

    arima_aic_search.csv und lstm_hpo_results.csv werden vom Notebook ohne
    Index-Spalte geschrieben (index=False); alle übrigen mit einer
    aussagekräftigen Index-Spalte (Datum bzw. Modell-/Reihenname).
    """
    path = RESULTS_DIR / name
    if not path.is_file():
        return None
    index_col = None if name in ("arima_aic_search.csv", "lstm_hpo_results.csv") else 0
    return pd.read_csv(path, index_col=index_col)


def best_arima_order() -> tuple[tuple, tuple] | None:
    """Liest die beste ARIMA/SARIMA-Ordnung aus einem Notebook-Lauf, falls vorhanden."""
    df = load_results_csv("arima_aic_search.csv")
    if df is None or len(df) == 0:
        return None
    best = df.sort_values("AIC").iloc[0]
    try:
        order = tuple(json.loads(best["order"].replace("(", "[").replace(")", "]")))
        seasonal = tuple(json.loads(best["seasonal"].replace("(", "[").replace(")", "]")))
        return order, seasonal
    except Exception:
        return None


def results_available() -> bool:
    return RESULTS_DIR.is_dir() and any(RESULTS_DIR.glob("*.csv"))
