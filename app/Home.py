"""Landing page der Streamlit-App: Überblick + Statusprüfung.

Optisch an docs/index.html (GitHub-Pages-Landingpage) angelehnt: gleiche
Farbpalette/Typografie, Ticker-Badge, Glass-Karten und ein echter
Live-Kurschart statt der reinen Text-Statusanzeige.
"""
import plotly.graph_objects as go
import streamlit as st

from common import RESULTS_DIR, TICKER, load_live_price_data, results_available

st.set_page_config(page_title="NVDA Kursprognose", page_icon="📈", layout="wide")

ACCENT = "#8fef00"
ACCENT_DIM = "#76b900"
CYAN = "#29e0ff"
GOOD = "#3ddc84"
BAD = "#ff5470"
BG = "#05070a"
SURFACE = "rgba(255,255,255,0.035)"
BORDER = "rgba(255,255,255,0.09)"
BORDER_ACCENT = "rgba(143,239,0,0.35)"
TEXT_MUTED = "#8b93a1"

st.html(
    f"""<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  html, body, [class*="css"] {{ font-family: 'JetBrains Mono', monospace; }}
  h1, h2, h3 {{ font-family: 'Space Grotesk', sans-serif !important; }}
  .ticker-badge {{
    display: inline-flex; align-items: center; gap: 0.5rem;
    font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; font-weight: 600;
    letter-spacing: 0.08em; color: {ACCENT};
    background: {SURFACE}; border: 1px solid {BORDER_ACCENT};
    padding: 0.35rem 0.8rem 0.35rem 0.6rem; border-radius: 999px; margin-bottom: 1rem;
  }}
  .ticker-badge .dot {{
    width: 7px; height: 7px; border-radius: 50%; background: {ACCENT};
    box-shadow: 0 0 8px 1px {ACCENT}; animation: pulse 2s ease-in-out infinite;
  }}
  @keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.35; }} }}
  .hero-title {{
    font-family: 'Space Grotesk', sans-serif; font-weight: 700;
    font-size: clamp(1.9rem, 4vw, 2.7rem); line-height: 1.1;
    letter-spacing: -0.02em; margin: 0 0 0.6rem;
  }}
  .hero-title .accent-text {{
    background: linear-gradient(100deg, {ACCENT} 10%, {CYAN} 90%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
  }}
  .status-card {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 12px;
    padding: 1.1rem 1.3rem; height: 100%;
  }}
  .status-card h4 {{
    font-family: 'Space Grotesk', sans-serif; font-size: 0.85rem;
    text-transform: uppercase; letter-spacing: 0.05em; color: {TEXT_MUTED};
    margin: 0 0 0.6rem; font-weight: 600;
  }}
  .status-pill {{
    display: inline-block; font-family: 'JetBrains Mono', monospace;
    font-size: 0.88rem; font-weight: 600; padding: 0.2rem 0.65rem;
    border-radius: 999px; margin-bottom: 0.5rem;
  }}
  .status-good {{ color: {GOOD}; background: rgba(61,220,132,0.14); }}
  .status-warn {{ color: #ffcc66; background: rgba(255,204,102,0.14); }}
  .status-bad {{ color: {BAD}; background: rgba(255,84,112,0.14); }}
  .status-card .caption {{ color: {TEXT_MUTED}; font-size: 0.8rem; }}
</style>"""
)

st.html(
    """<span class="ticker-badge"><span class="dot"></span>NVDA · NASDAQ · WEB-APP</span>
<div class="hero-title">NVIDIA-Kursprognose — <span class="accent-text">Web-App</span></div>"""
)

st.markdown(
    """
Begleit-App zum Notebook [`NVIDIA_Kursprognose_LSTM.ipynb`](https://github.com/I-Fichtner-I/Stock-Price-Prediction/blob/main/NVIDIA_Kursprognose_LSTM.ipynb).
Zwei Ansichten stehen zur Verfügung (siehe Seitenleiste):

- **📊 Dashboard** — visualisiert die Ergebnisse eines Notebook-Laufs
  (Modellvergleich, Metriken, Trading-Simulation, ARIMA-/LSTM-Suche).
- **🔴 Live-Prognose** — lädt aktuelle Kursdaten und erstellt eine
  Prognose in Echtzeit (Naive Baseline + live gefittetes ARIMA).

> **Hinweis:** Diese App dient der Illustration und ist **keine
> Anlageberatung**. Details zur Methodik siehe [README](https://github.com/I-Fichtner-I/Stock-Price-Prediction#readme)
> · Landingpage: [i-fichtner-i.github.io/Stock-Price-Prediction](https://i-fichtner-i.github.io/Stock-Price-Prediction/).
"""
)

# ── Live-Chart ────────────────────────────────────────────────────────────────
with st.spinner("Lade Live-Kurschart..."):
    try:
        chart_df, chart_source = load_live_price_data(TICKER, start="2024-01-01")
    except Exception:
        chart_df, chart_source = None, None

if chart_df is not None and len(chart_df) > 1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=chart_df.index, y=chart_df["Close"],
        mode="lines", line=dict(color=ACCENT, width=2),
        fill="tozeroy", fillcolor="rgba(143,239,0,0.12)",
        hovertemplate="%{x|%Y-%m-%d}<br>$%{y:.2f}<extra></extra>",
    ))
    fig.update_layout(
        height=260, margin=dict(l=0, r=0, t=8, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, monospace", color=TEXT_MUTED, size=11),
        xaxis=dict(showgrid=False, showline=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor=BORDER, zeroline=False, tickprefix="$"),
        hovermode="x unified",
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

# ── Status-Karten ────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    if results_available():
        pill = '<span class="status-pill status-good">● Dashboard einsatzbereit</span>'
    else:
        pill = '<span class="status-pill status-warn">● Kein results/-Verzeichnis</span>'
    st.html(
        f"""<div class="status-card">
  <h4>Status · Notebook-Ergebnisse</h4>
  {pill}
  <div class="caption">Erwarteter Pfad: {RESULTS_DIR}</div>
</div>"""
    )

with col2:
    if chart_df is not None:
        label_map = {
            "yfinance": ("status-good", "● Live (Yahoo Finance)"),
            "csv_snapshot": ("status-warn", "● CSV-Snapshot (kein Live-Zugriff)"),
            "synthetic": ("status-bad", "● Synthetischer Ersatzdatensatz"),
        }
        css_class, label = label_map[chart_source]
        last_close = float(chart_df["Close"].iloc[-1])
        last_date = chart_df.index[-1].date()
        st.html(
            f"""<div class="status-card">
  <h4>Status · Live-Datenquelle</h4>
  <span class="status-pill {css_class}">{label}</span>
  <div class="caption">Letzter Kurs: ${last_close:.2f} USD ({last_date})</div>
</div>"""
        )
    else:
        st.html(
            """<div class="status-card">
  <h4>Status · Live-Datenquelle</h4>
  <span class="status-pill status-bad">● Datenquelle nicht verfügbar</span>
</div>"""
        )
