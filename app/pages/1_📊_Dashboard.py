"""Dashboard: visualisiert results/*.csv aus einem Notebook-Lauf."""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from common import RESULTS_DIR, load_results_csv, results_available

st.set_page_config(page_title="Dashboard — NVDA Kursprognose", page_icon="📊", layout="wide")
st.title("📊 Dashboard — Notebook-Ergebnisse")

if not results_available():
    st.warning(
        "Kein `results/`-Verzeichnis gefunden. Führe zuerst das Notebook "
        "(`NVIDIA_Kursprognose_LSTM.ipynb`) vollständig aus — es exportiert seine "
        "Ergebnisse automatisch nach `results/*.csv` (siehe README, Abschnitt 8)."
    )
    st.stop()

# ── 1) Prognosevergleich ─────────────────────────────────────────────────────
st.header("1  Ist-Kurs vs. Prognosen")
fc = load_results_csv("forecast_comparison.csv")
if fc is not None:
    fc.index = pd.to_datetime(fc.index)
    fig = go.Figure()
    colors = {"Ist": "black", "Naiv": "steelblue", "MovingAverage": "gray",
              "ARIMA_SARIMA": "green", "LSTM_Level": "crimson", "LSTM_LogRendite": "darkorange"}
    for col in fc.columns:
        fig.add_trace(go.Scatter(x=fc.index, y=fc[col], name=col,
                                  line=dict(color=colors.get(col), width=1.6 if col == "Ist" else 1.1)))
    fig.update_layout(height=500, hovermode="x unified",
                       xaxis=dict(rangeslider=dict(visible=True)),
                       yaxis_title="Close (USD)")
    st.plotly_chart(fig, width='stretch')
else:
    st.info("`forecast_comparison.csv` nicht gefunden.")

# ── 2) Metriken ───────────────────────────────────────────────────────────────
st.header("2  Modellvergleich (Testmenge)")
metrics = load_results_csv("metrics_comparison.csv")
if metrics is not None:
    metrics_sorted = metrics.sort_values("RMSE")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(
            metrics_sorted.style.format({c: "{:.4f}" for c in metrics_sorted.columns})
            .highlight_min(subset=["RMSE", "MAE"], color="#c6f6d5")
            .highlight_max(subset=["R²"], color="#c6f6d5"),
            width='stretch',
        )
    with col2:
        best = metrics_sorted.index[0]
        st.metric("Bestes Modell (RMSE)", best, f"{metrics_sorted.loc[best, 'RMSE']:.3f} USD")
        if "Theil_U" in metrics_sorted.columns:
            u = metrics_sorted.loc[best, "Theil_U"]
            st.metric("Theil's U", f"{u:.4f}", "besser als naiv" if u < 1 else "nicht besser als naiv")
    bar = go.Figure()
    bar.add_trace(go.Bar(x=metrics_sorted.index, y=metrics_sorted["RMSE"], marker_color="steelblue"))
    bar.update_layout(height=350, title="RMSE je Modell (USD)", yaxis_title="RMSE")
    st.plotly_chart(bar, width='stretch')
else:
    st.info("`metrics_comparison.csv` nicht gefunden.")

# ── 3) ARIMA-Suche ────────────────────────────────────────────────────────────
st.header("3  ARIMA/SARIMA — Automatische Ordnungswahl")
aic = load_results_csv("arima_aic_search.csv")
if aic is not None:
    st.dataframe(aic.sort_values("AIC").head(10), width='stretch')
else:
    st.info("`arima_aic_search.csv` nicht gefunden.")

# ── 4) LSTM-HPO ───────────────────────────────────────────────────────────────
st.header("4  LSTM-Hyperparameter-Suche")
hpo = load_results_csv("lstm_hpo_results.csv")
if hpo is not None:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(hpo.sort_values("Val_RMSE"), width='stretch')
    with c2:
        hpo_sorted = hpo.sort_values("Val_RMSE", ascending=False)
        labels = hpo_sorted["Neuronen"].astype(str) + " · d" + hpo_sorted["Dropout"].astype(str)
        fig_hpo = go.Figure(go.Bar(x=hpo_sorted["Val_RMSE"], y=labels, orientation="h",
                                    marker_color="mediumpurple"))
        fig_hpo.update_layout(height=300, title="Validierungs-RMSE je Konfiguration",
                               xaxis_title="Val-RMSE (USD)")
        st.plotly_chart(fig_hpo, width='stretch')
else:
    st.info("`lstm_hpo_results.csv` nicht gefunden (kein TensorFlow beim Notebook-Lauf).")

hpo_r = load_results_csv("lstm_log_hpo_results.csv")
if hpo_r is not None:
    st.subheader("4b  LSTM-HPO — Log-Renditen-Modell (Kap. 6.5, eigene Suche)")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(hpo_r.sort_values("Val_RMSE"), width='stretch')
    with c2:
        hpo_r_sorted = hpo_r.sort_values("Val_RMSE", ascending=False)
        labels_r = hpo_r_sorted["Neuronen"].astype(str) + " · d" + hpo_r_sorted["Dropout"].astype(str)
        fig_hpo_r = go.Figure(go.Bar(x=hpo_r_sorted["Val_RMSE"], y=labels_r, orientation="h",
                                      marker_color="darkorange"))
        fig_hpo_r.update_layout(height=300, title="Validierungs-RMSE je Konfiguration (Log-Rendite)",
                                 xaxis_title="Val-RMSE (USD)")
        st.plotly_chart(fig_hpo_r, width='stretch')

# ── 5) Trading-Simulation ──────────────────────────────────────────────────────
st.header("5  Trading-Simulation")
strat = load_results_csv("trading_simulation.csv")
if strat is not None:
    st.dataframe(
        strat.style.format({"Kum_Rendite": "{:.2%}", "Sharpe": "{:.3f}", "MaxDrawdown": "{:.2%}"}),
        width='stretch',
    )
else:
    st.info("`trading_simulation.csv` nicht gefunden.")

# ── 6) Stationarität ────────────────────────────────────────────────────────────
st.header("6  Stationaritätstests (ADF + KPSS)")
stat = load_results_csv("stationarity_tests.csv")
if stat is not None:
    st.dataframe(stat, width='stretch')
else:
    st.info("`stationarity_tests.csv` nicht gefunden.")

# ── 7) Interaktive Plotly-Grafik aus dem Notebook (falls vorhanden) ────────────
html_path = RESULTS_DIR / "interactive_forecast.html"
if html_path.is_file():
    st.header("7  Interaktive Notebook-Grafik (Kap. 7.2d)")
    st.components.v1.html(html_path.read_text(encoding="utf-8"), height=650, scrolling=True)
