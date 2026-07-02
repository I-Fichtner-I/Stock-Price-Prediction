"""Landing page der Streamlit-App: Überblick + Statusprüfung."""
import streamlit as st

from common import RESULTS_DIR, TICKER, load_live_price_data, results_available

st.set_page_config(page_title="NVDA Kursprognose", page_icon="📈", layout="wide")

st.title("📈 NVIDIA-Kursprognose — Web-App")
st.markdown(
    """
Begleit-App zum Notebook [`NVIDIA_Kursprognose_LSTM.ipynb`](../NVIDIA_Kursprognose_LSTM.ipynb).
Zwei Ansichten stehen zur Verfügung (siehe Seitenleiste):

- **📊 Dashboard** — visualisiert die Ergebnisse eines Notebook-Laufs
  (Modellvergleich, Metriken, Trading-Simulation, ARIMA-/LSTM-Suche).
- **🔴 Live-Prognose** — lädt aktuelle Kursdaten und erstellt eine
  Prognose in Echtzeit (Naive Baseline + live gefittetes ARIMA).

> **Hinweis:** Diese App dient der Illustration und ist **keine
> Anlageberatung**. Details zur Methodik siehe [README](../README.md).
"""
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Status: Notebook-Ergebnisse")
    if results_available():
        st.success(f"`results/`-Verzeichnis gefunden — Dashboard ist einsatzbereit.")
    else:
        st.warning(
            "Kein `results/`-Verzeichnis gefunden. Führe zuerst das Notebook aus "
            "(siehe README, Abschnitt 8), damit das Dashboard Daten anzeigen kann."
        )
    st.caption(f"Erwarteter Pfad: `{RESULTS_DIR}`")

with col2:
    st.subheader("Status: Live-Datenquelle")
    with st.spinner("Prüfe Datenquelle..."):
        try:
            df, source = load_live_price_data(TICKER, start="2024-01-01")
            label = {"yfinance": "✅ Live (Yahoo Finance)",
                      "csv_snapshot": "🟡 CSV-Snapshot (kein Live-Zugriff)",
                      "synthetic": "⚠️ Synthetischer Ersatzdatensatz"}[source]
            st.info(label)
            st.caption(f"Letzter Kurs: {df['Close'].iloc[-1]:.2f} USD "
                       f"({df.index[-1].date()})")
        except Exception as e:
            st.error(f"Datenquelle nicht verfügbar: {e}")
