"""Live-Prognose: aktuelle Kursdaten laden und in Echtzeit prognostizieren.

Nutzt Naive Baseline + ein live gefittetes ARIMA/SARIMA (statsmodels, wenige
Sekunden Laufzeit) statt des LSTM aus dem Notebook: Das LSTM-Training dauert
zu lange für eine interaktive Web-App, und laut Notebook (Kap. 8.3) bietet es
ohnehin keinen robusten Vorteil gegenüber ARIMA/Naiv auf dieser Datenbasis.
"""
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import add_technical_features, best_arima_order, load_live_price_data

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Live-Prognose — NVDA", page_icon="🔴", layout="wide")
st.title("🔴 Live-Prognose")
st.caption(
    "Lädt aktuelle Kursdaten und erstellt eine 1-Schritt-Prognose in Echtzeit. "
    "**Keine Anlageberatung.**"
)

with st.sidebar:
    st.header("Einstellungen")
    ticker = st.text_input("Ticker", value="NVDA").strip().upper()
    lookback_days = st.slider("Historie (Handelstage) für den Chart", 60, 500, 180)
    run = st.button("🔄 Aktualisieren", type="primary")

if "live_data" not in st.session_state or run:
    with st.spinner(f"Lade {ticker}-Kursdaten..."):
        try:
            df, source = load_live_price_data(ticker, start="2018-01-01")
            st.session_state["live_data"] = (df, source)
        except Exception as e:
            st.error(f"Datenabruf fehlgeschlagen: {e}")
            st.stop()

df, source = st.session_state["live_data"]
source_label = {"yfinance": "✅ Live (Yahoo Finance)",
                 "csv_snapshot": "🟡 CSV-Snapshot (kein Live-Zugriff)",
                 "synthetic": "⚠️ Synthetischer Ersatzdatensatz — nur Illustration"}[source]
st.info(f"Datenquelle: {source_label}  ·  Stand: {df.index[-1].date()}  "
        f"·  {len(df):,} Handelstage geladen")

feat = add_technical_features(df).dropna()
last_close = float(feat["Close"].iloc[-1])
last_date = feat.index[-1]

# ── Naive Baseline ────────────────────────────────────────────────────────────
naive_forecast = last_close

# ── Live-ARIMA (schnell, statsmodels) ─────────────────────────────────────────
arima_forecast, arima_lo, arima_hi, arima_order_used = None, None, None, None
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    order_seasonal = best_arima_order()
    order = order_seasonal[0] if order_seasonal else (1, 1, 1)
    seasonal = order_seasonal[1] if order_seasonal else (0, 0, 0, 0)
    arima_order_used = (order, seasonal)

    log_close = np.log(feat["Close"].values.astype(float))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SARIMAX(log_close, order=order, seasonal_order=seasonal,
                       enforce_stationarity=False, enforce_invertibility=False).fit(disp=0)
        fc = res.get_forecast(steps=1)
        arima_forecast = float(np.exp(fc.predicted_mean[0]))
        ci = np.exp(fc.conf_int(alpha=0.20)[0])   # 80 %-Intervall
        arima_lo, arima_hi = float(ci[0]), float(ci[1])
except Exception as e:
    st.warning(f"Live-ARIMA-Fit fehlgeschlagen, zeige nur Naive Baseline. ({e})")

# ── Kennzahlen ────────────────────────────────────────────────────────────────
st.header(f"Prognose für den nächsten Handelstag nach {last_date.date()}")
c1, c2, c3 = st.columns(3)
c1.metric("Letzter Schlusskurs", f"{last_close:.2f} USD")
c2.metric("Naive Prognose (P_t)", f"{naive_forecast:.2f} USD", "Δ 0.00 (per Definition)")
if arima_forecast is not None:
    delta = arima_forecast - last_close
    c3.metric(f"ARIMA{arima_order_used[0]} Prognose", f"{arima_forecast:.2f} USD",
              f"{delta:+.2f} USD")
    st.caption(f"80 %-Prognoseintervall (ARIMA): [{arima_lo:.2f}, {arima_hi:.2f}] USD  ·  "
               f"Ordnung: ARIMA{arima_order_used[0]} × SARIMA{arima_order_used[1]}"
               + ("  (aus Notebook-Lauf übernommen)" if best_arima_order() else "  (Standardordnung)"))
else:
    c3.metric("ARIMA Prognose", "n/a")

st.warning(
    "⚠️ Diese Prognose basiert auf reinen historischen Kursdaten (keine "
    "Fundamentaldaten, News, Makro-Indikatoren) und dient ausschließlich der "
    "wissenschaftlichen Illustration — **keine Handelsempfehlung**. Laut "
    "Notebook-Analyse (Kap. 8) liegt die Richtungsgenauigkeit datengetriebener "
    "Modelle auf dieser Basis nahe am Zufall (50 %)."
)

# ── Chart ─────────────────────────────────────────────────────────────────────
st.header("Kursverlauf mit Prognose")
plot_df = feat.tail(lookback_days)
fig = go.Figure()
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Close"], name="Close",
                          line=dict(color="black", width=1.4)))
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["MA20"], name="MA20",
                          line=dict(color="steelblue", width=1, dash="dot")))
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["MA50"], name="MA50",
                          line=dict(color="darkorange", width=1, dash="dot")))

next_bday = pd.bdate_range(last_date, periods=2)[-1]

fig.add_trace(go.Scatter(x=[next_bday], y=[naive_forecast], name="Naiv-Prognose",
                          mode="markers", marker=dict(color="steelblue", size=11, symbol="diamond")))
if arima_forecast is not None:
    fig.add_trace(go.Scatter(x=[next_bday], y=[arima_forecast], name="ARIMA-Prognose",
                              mode="markers", marker=dict(color="green", size=11, symbol="star")))
    fig.add_trace(go.Scatter(x=[next_bday, next_bday], y=[arima_lo, arima_hi],
                              name="ARIMA 80%-Intervall", mode="lines",
                              line=dict(color="green", width=6), opacity=0.3))

fig.update_layout(height=520, hovermode="x unified", yaxis_title="USD",
                   xaxis=dict(rangeslider=dict(visible=True)))
st.plotly_chart(fig, width='stretch')

# ── Technische Indikatoren ───────────────────────────────────────────────────
with st.expander("Technische Indikatoren (aktueller Stand)"):
    latest = feat.iloc[-1]
    ic1, ic2, ic3, ic4 = st.columns(4)
    ic1.metric("RSI14", f"{latest['RSI14']:.1f}",
               "überkauft" if latest["RSI14"] > 70 else ("überverkauft" if latest["RSI14"] < 30 else "neutral"))
    ic2.metric("MACD", f"{latest['MACD']:.3f}")
    ic3.metric("Volatilität (20T)", f"{latest['Volatility20']:.4f}")
    ic4.metric("BB %B", f"{latest['BB_Pct']:.2f}")
