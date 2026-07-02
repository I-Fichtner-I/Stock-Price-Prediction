# NVDA-Kursprognose — Streamlit-App

Begleit-App zum Notebook [`NVIDIA_Kursprognose_LSTM.ipynb`](../NVIDIA_Kursprognose_LSTM.ipynb).
Zwei Seiten:

- **📊 Dashboard** — visualisiert die `results/*.csv`-Ergebnisse eines
  Notebook-Laufs (Modellvergleich, Metriken, Trading-Simulation,
  ARIMA-/LSTM-Suche). Erfordert, dass das Notebook zuvor einmal vollständig
  ausgeführt wurde (siehe [README](../README.md), Abschnitt 8).
- **🔴 Live-Prognose** — lädt aktuelle Kursdaten (Live-yfinance, sonst
  CSV-Snapshot, sonst synthetischer Ersatzdatensatz — dieselbe 3-Stufen-Logik
  wie im Notebook) und erstellt eine 1-Schritt-Prognose in Echtzeit: Naive
  Baseline + ein schnell live gefittetes ARIMA/SARIMA. Das LSTM aus dem
  Notebook wird hier bewusst **nicht** verwendet — Training dauert zu lange
  für eine interaktive App, und laut Notebook-Analyse (Kap. 8.3) bietet es
  ohnehin keinen robusten Vorteil gegenüber ARIMA/Naiv.

## Installation & Start

```bash
# Aus dem Repository-Root:
pip install -r requirements.txt        # Kernabhängigkeiten (pandas, yfinance, statsmodels, ...)
pip install -r app/requirements.txt    # Streamlit + Plotly

streamlit run app/Home.py
```

Die App öffnet sich im Browser (Standard: http://localhost:8501).

## Hinweise

- **Keine Anlageberatung.** Beide Seiten dienen ausschließlich der
  wissenschaftlichen Illustration der im Notebook entwickelten Methodik.
- Das Dashboard liest `results/*.csv` relativ zum Repository-Root — diese
  Dateien werden nicht mitversioniert (siehe `.gitignore`), sondern bei
  Notebook-Ausführung neu erzeugt.
- Die Live-Prognose funktioniert auch ohne Netzwerkzugriff (Fallback auf
  `data/nvda_ohlcv.csv` bzw. synthetische Daten), liefert dann aber keine
  aktuellen Werte.
