# Prognose von Aktienkursen mittels maschinellen Lernens
## Am Beispiel der NVIDIA-Aktie (LSTM, ARIMA/SARIMA, Baselines)

> **Seminar-/Termpaper-Notebook** · Reproduzierbares Python-Experiment ·
> Zeitreihenanalyse & Deep Learning

---

## Inhaltsverzeichnis

1. [Forschungsfrage & Hypothesen](#1-forschungsfrage--hypothesen)
2. [Datenbasis](#2-datenbasis)
3. [Methodologie](#3-methodologie)
4. [Modellarchitektur](#4-modellarchitektur)
5. [Evaluationsrahmen](#5-evaluationsrahmen)
6. [Zentrale Ergebnisse](#6-zentrale-ergebnisse)
7. [Explainability (SHAP)](#7-explainability-shap)
8. [Reproduzierbarkeit & Ausführung](#8-reproduzierbarkeit--ausführung)
9. [Projektstruktur](#9-projektstruktur)
10. [Abhängigkeiten](#10-abhängigkeiten)

---

## 1  Forschungsfrage & Hypothesen

**Zentrale Frage:** Kann ein LSTM-Netz den Schlusskurs der NVIDIA-Aktie mit
messbarem Mehrwert gegenüber naiven Referenzmodellen prognostizieren?

| # | Hypothese |
|---|-----------|
| H₁ | LSTM übertrifft die naive Baseline (Yesterday's Close) gemessen an RMSE und MAE. |
| H₂ | ARIMA/SARIMA schneidet besser ab als die naive Baseline, aber schwächer als LSTM. |
| H₃ | Die Richtungsgenauigkeit (Directional Accuracy) des LSTM übersteigt 50 % signifikant. |
| H₄ | SHAP-Werte zeigen, dass kurzfristige Lag-Features den größten Erklärungsbeitrag leisten. |

> **Methodischer Hinweis:** Aufgrund des bekannten *Lag-Phänomens* bei
> Niveau-basierten LSTM-Prognosen (das Modell lernt primär $\hat{y}_t \approx y_{t-1}$)
> werden in Kapitel 6.5 ergänzend Log-Renditen als stationäre Zielvariable getestet.

---

## 2  Datenbasis

```
Ticker    : NVDA (NVIDIA Corporation, NASDAQ)
Zeitraum  : 2019-01-01 – 2024-12-31  (ca. 1 510 Handelstage)
Quelle    : Yahoo Finance via yfinance  |  Fallback: synthetischer GBM-Datensatz
Frequenz  : Täglich (Schlusskurse, OHLCV)
Währung   : USD
```

### Rohdatenstruktur

| Spalte | Typ | Beschreibung |
|--------|-----|-------------|
| `Date` | datetime64 | Handelstag (NYSE-Kalender) |
| `Open` | float64 | Eröffnungskurs |
| `High` | float64 | Tageshoch |
| `Low` | float64 | Tagestief |
| `Close` | float64 | Schlusskurs *(Zielvariable)* |
| `Volume` | int64 | Handelsvolumen (Stückzahl) |
| `Adj Close` | float64 | Dividenden-/Split-bereinigter Schlusskurs |

### Datenqualitätsprüfung

- Fehlende Werte: Vorwärts-Imputation (`ffill`) nach Börsentagen
- Ausreißer: IQR-Verfahren, keine Entfernung (Kurssprünge sind ökonomisch relevant)
- Split-Bereinigung: `yfinance` liefert bereits adjustierte Daten

---

## 3  Methodologie

### 3.1  Untersuchungsdesign

```
┌─────────────────────────────────────────────────────────────────┐
│                     UNTERSUCHUNGSABLAUF                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Rohdaten (yfinance)                                            │
│       │                                                         │
│       ▼                                                         │
│  Datenqualitätsprüfung & Cleaning                               │
│       │                                                         │
│       ▼                                                         │
│  Explorative Datenanalyse (EDA)                                 │
│   ├── Deskriptive Statistik & Verteilungsanalyse                │
│   ├── ADF-Test + KPSS-Test (Stationarität)                      │
│   ├── ACF / PACF  (Autokorrelationsstruktur)                    │
│   ├── Renditeanalyse (Log-Renditen, Volatilität)                │
│   └── Ereignisanalyse (Kurssprünge >= |5 %|)                    │
│       │                                                         │
│       ▼                                                         │
│  Feature Engineering                                            │
│   ├── Technische Indikatoren (MA5/MA20/MA60, EMA, BB, RSI)      │
│   ├── Lag-Features (t-1 ... t-5)                                │
│   ├── Rendite- & Volatilitätsmerkmale                           │
│   └── Kalendermerkmale (Wochentag, Monat)                       │
│       │                                                         │
│       ▼                                                         │
│  Chronologischer Train / Val / Test-Split (70 / 15 / 15 %)      │
│       │                                                         │
│       ▼                                                         │
│  MinMaxScaler  (kalibriert ausschliesslich auf Trainingsdaten)  │
│       │                                                         │
│       ▼                                                         │
│  Sliding-Window-Transformation  (Fensterlänge W = 60 Tage)      │
│       │                                                         │
│       ├──────────────────────────────┐                          │
│       ▼                              ▼                          │
│  Statistische Modelle            LSTM-Modell                    │
│       ├── Naive Baseline             ├── Architektur-Suche      │
│       ├── Moving-Average-Baseline    │   (Schichten, Neuronen,  │
│       └── ARIMA/SARIMA (AIC-Suche)   │    Dropout, Lernrate)    │
│                                      └── Walk-Forward-CV        │
│       │                              │                          │
│       └──────────────┬───────────────┘                          │
│                      ▼                                          │
│  Evaluation (MAE, RMSE, MAPE, sMAPE, Theil's U, DA)             │
│                      │                                          │
│                      ▼                                          │
│  Explainability (SHAP / Permutationswichtigkeit)                │
│                      │                                          │
│                      ▼                                          │
│  Trading-Simulation (Signalstrategie vs. Buy-and-Hold)          │
│                      │                                          │
│                      ▼                                          │
│  Diskussion & Fazit                                             │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2  Train/Val/Test-Aufteilung

```
Gesamtzeitraum: ca. 1 510 Handelstage
────────────────────────────────────────────────────────────────────────────
│<──────── Training (70 %) ────────>│<── Val (15 %) ──>│<── Test (15 %) ──>│
│  ca. 1 057 Tage                   │   ca. 227 Tage   │     ca. 226 Tage  │
│  (2019-01 – 2022-08)              │   (2022-08 –     │     (2023-04 –    │
│                                   │    2023-04)      │      2024-12)     │
────────────────────────────────────────────────────────────────────────────
```

> **Keine Data-Leakage-Risiken:** Scaler-Kalibrierung, Feature-Berechnung und
> Hyperparameter-Selektion erfolgen ausschliesslich auf Trainings- bzw.
> Validierungsdaten. Der Testset wird erst zur finalen Evaluation verwendet.

### 3.3  Sliding-Window-Transformation

Aus der skalierten Zeitreihe werden Samples der Form

```
(X_t, y_t) = ([x_{t-W}, ..., x_{t-1}],  x_t)
```

erzeugt, wobei `W = 60` Handelstage (ca. 3 Monate) als Look-back-Fenster gewählt
wurde. Jedes Sample `X_t` hat die Form `(60, F)` mit `F = 13` Features.

---

## 4  Modellarchitektur

### 4.1  LSTM-Netzwerk

```
Eingabeschicht
  │  Shape: (Batch, W=60, F=13)   [60 Zeitschritte x 13 Features]
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│  LSTM-Schicht 1                                                 │
│    Einheiten: {32, 64, 128}  (Hyperparameter-Suche)             │
│    return_sequences: True  (falls mehrstufig)                   │
│    Dropout: {0.0, 0.2, 0.3}                                     │
├─────────────────────────────────────────────────────────────────┤
│  [Optional] LSTM-Schicht 2                                      │
│    Einheiten: wie Schicht 1                                     │
│    return_sequences: False                                      │
│    Dropout: wie Schicht 1                                       │
├─────────────────────────────────────────────────────────────────┤
│  [Optional] LSTM-Schicht 3                                      │
│    Einheiten: wie Schicht 1                                     │
└─────────────────────────────────────────────────────────────────┘
  │
  ▼
Dense (1 Neuron, linear)   -->   y_hat_t  (skalierter Schlusskurs)
  │
  ▼
Inverse MinMax-Transformation   -->   Preisprognose in USD
```

**LSTM-Zellmechanik:**

```
f_t = sigmoid(W_f * [h_{t-1}, x_t] + b_f)     [Forget Gate]
i_t = sigmoid(W_i * [h_{t-1}, x_t] + b_i)     [Input Gate]
c~_t = tanh(W_c * [h_{t-1}, x_t] + b_c)       [Cell Candidate]
c_t = f_t * c_{t-1} + i_t * c~_t              [Cell State]
o_t = sigmoid(W_o * [h_{t-1}, x_t] + b_o)     [Output Gate]
h_t = o_t * tanh(c_t)                          [Hidden State]
```

### 4.2  Hyperparameter-Suchraum (LSTM)

| Parameter | Wertebereich | Auswahlkriterium |
|-----------|-------------|-----------------|
| Anzahl LSTM-Schichten | {1, 2, 3} | Validierungs-RMSE |
| Neuronen pro Schicht | {32, 64, 128} | Validierungs-RMSE |
| Dropout-Rate | {0.0, 0.2, 0.3} | Validierungs-RMSE |
| Lernrate | {0.001, 0.0005} | Validierungs-RMSE |
| Batch-Grösse | 32 | fest |
| Max. Epochen | 100 | EarlyStopping (Patience=10) |
| Optimierer | Adam | fest |
| Verlustfunktion | MSE | fest |

### 4.3  ARIMA/SARIMA — Automatische Ordnungswahl

```
Stufe 1 — Nicht-saisonal (AIC-Gittersuche):
  p in {0,1,2,3,4}  x  d in {1,2}  x  q in {0,1,2,3,4}
  --> 50 Kandidaten, Auswahl nach minimalem AIC

Stufe 2 — Saisonal (Verfeinerung):
  (P,D,Q)_m  mit m = 5  (Wochenperiodizität)
  P,Q in {0,1}  x  D in {0,1}
  --> Beste SARIMA-Erweiterung nach AIC

Prognose: Walk-Forward (1-step-ahead, Rolling Refit)
```

---

## 5  Evaluationsrahmen

### 5.1  Metriken

| Metrik | Formel | Einheit | Interpretation |
|--------|--------|---------|---------------|
| **MAE** | mean(\|y_t - y_hat_t\|) | USD | Mittlerer absoluter Fehler |
| **RMSE** | sqrt(mean((y_t - y_hat_t)²)) | USD | Bestraft Ausreisser stärker |
| **MAPE** | 100 * mean(\|(y_t - y_hat_t) / y_t\|) | % | Skalenunabhängig |
| **sMAPE** | 200 * mean(\|y_t - y_hat_t\| / (\|y_t\| + \|y_hat_t\|)) | % | Robuster bei kleinen y_t |
| **R²** | 1 - SS_res / SS_tot | — | Trügerisch hoch bei Trendreihen |
| **Theil's U** | RMSE_Modell / RMSE_Naiv | — | < 1: besser als naiv |
| **DA** | mean(1[sgn(Δy_hat_t) = sgn(Δy_t)]) | % | Richtungsgenauigkeit |

### 5.2  Walk-Forward-Kreuzvalidierung

```
Zeitreihen-CV mit expandierendem Trainingsfenster (3 Folds, siehe CV_FOLDS in
Kap. 6.4 des Notebooks — bewusst klein gehalten, um die interaktive
Ausführbarkeit zu gewährleisten):

Fold 1:  [████████████████████████][░░░░░░░░░░░░]
Fold 2:  [████████████████████████████][░░░░░░░░░░░░]
Fold 3:  [████████████████████████████████][░░░░░░░░░░░░]

         [████] Training   [░░░░] Test (je Fold)
```

---

## 6  Zentrale Ergebnisse

> Die nachfolgenden Werte sind **laufzeitabhängig** (echte Yahoo-Finance-Daten
> vs. synthetischer Fallback, gewählte Hyperparameter). Alle Zahlen werden bei
> Ausführung des Notebooks reproduzierbar in `results/` als CSV gespeichert.

### 6.1  Modellvergleich (Testmenge)

<!-- AUTO-GENERATED:METRICS_TABLE:START -->
```
Modell               │ MAE (USD) │ RMSE (USD) │ MAPE (%) │      R² │ Theil's U │ DA (%)
─────────────────────┼───────────┼────────────┼──────────┼─────────┼───────────┼────────
Naive Baseline       │     2.029 │      2.927 │     2.24 │  0.9936 │    1.0000 │      –
Moving Average       │     5.645 │      7.539 │     6.36 │  0.9575 │    2.5754 │      –
ARIMA/SARIMA         │     2.060 │      2.952 │     2.27 │  0.9935 │    1.0085 │      –
LSTM (Niveau)        │    11.746 │     16.531 │    10.44 │  0.7955 │    5.6469 │      –
LSTM (Log-Renditen)  │     2.267 │      3.302 │     2.44 │  0.9918 │    1.1280 │      –
─────────────────────┴───────────┴────────────┴──────────┴─────────┴───────────┴────────
Automatisch befüllt aus results/metrics_comparison.csv (2026-07-02)
```
<!-- AUTO-GENERATED:METRICS_TABLE:END -->

*Wird automatisch aus `results/metrics_comparison.csv` befüllt — siehe
[`_update_readme_results.py`](.github/_update_readme_results.py) (nach Notebook-Ausführung
ausführen).*

### 6.2  Schlüsselbefund: Lag-Phänomen

Ein typisches Problem bei Niveau-basierten LSTM-Prognosen ist das **Lag-Phänomen** —
das Modell lernt näherungsweise `y_hat_t ≈ y_{t-1}`:

```
Kurs (USD)
    ^
600 │                          /--\   -- Tatsächlich
    │                      /--/    \
500 │               /------/        \--
    │           /---/
400 │       /---/       /--\   ....  LSTM-Prognose (verschoben)
    │   /---/       /---/    \
300 │---/       ----/         \------
    └────────────────────────────────────────> Zeit
                   ^
                   Lag ca. 1-2 Tage
```

Das Modell erzeugt Theil's U nahe 1.0, liefert also kaum Mehrwert gegenüber
der naiven Baseline. Kapitel 6.5 adressiert dies durch Prognose auf
**Log-Renditen** als stationäre Zielvariable:

```
r_t = ln(P_t / P_{t-1})
```

### 6.3  Stationaritätsanalyse (ADF + KPSS)

<!-- AUTO-GENERATED:STATIONARITY_TABLE:START -->
| Zeitreihe | ADF p-Wert | KPSS-Befund | Schluss |
|-----------|-----------|-------------|---------|
| Close (Niveau) | 1 | stationär abgelehnt | **nicht stationär** |
| Log-Renditen | < 0.001 | stationär | **stationär** |
| Einfache Renditen | < 0.001 | stationär | **stationär** |
<!-- AUTO-GENERATED:STATIONARITY_TABLE:END -->

*Wird automatisch aus `results/stationarity_tests.csv` befüllt — siehe
[`_update_readme_results.py`](.github/_update_readme_results.py) (nach Notebook-Ausführung
ausführen).*

---

## 7  Explainability (SHAP)

### 7.1  Merkmalswichtigkeit

Mit **SHAP (SHapley Additive exPlanations)** werden die Eingangsmerkmale nach
ihrem durchschnittlichen absoluten Shapley-Wert geordnet:

```
Feature-Importance (mean |SHAP|, normiert; schematisch):

Close_lag1    ████████████████████  hoch
Close_lag2    ██████████████
MA_20         ████████████
MA_60         ██████████
Volatility    ███████
RSI_14        █████
Volume        ████
MA_5          ████
EMA_12        ██
Weekday       ██                    gering

Echte Werte: results/shap_feature_importance.csv
```

### 7.2  Zeitschritt-Wichtigkeit

Die SHAP-Analyse über das 60-Tage-Fenster zeigt: Die letzten 1–5 Handelstage
tragen den grössten Anteil zur Prognose bei:

```
Zeitschritt-Wichtigkeit im 60-Tage-Fenster (schematisch):

t-1   ████████████████████████  (höchster Beitrag)
t-2   ██████████████████
t-3   █████████████
t-4   █████████
t-5   ███████
...   ████
t-60  ██                        (geringster Beitrag)
```

> Fehlt das `shap`-Paket, berechnet das Notebook automatisch eine
> **modell-agnostische Permutationswichtigkeit** als äquivalenten Fallback.

---

## 8  Reproduzierbarkeit & Ausführung


### Voraussetzungen


- Python >= 3.10
- Jupyter Notebook/Lab >= 7.0


### Installation & Start


```bash
# 1) Repository klonen
git clone https://github.com/I-Fichtner-I/stock-price-prediction.git
cd stock-price-prediction

# 2) Virtuelle Umgebung einrichten (empfohlen)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3) Kernabhängigkeiten installieren
pip install -r requirements.txt

# 4) Optional: LSTM + Explainability
pip install "tensorflow-cpu>=2.16.0" "shap>=0.45.0"

# 5a) Interaktiv im Browser starten
jupyter notebook NVIDIA_Kursprognose_LSTM.ipynb

# 5b) Headless ausführen (alle Zellen neu rechnen)
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1200 \
  NVIDIA_Kursprognose_LSTM.ipynb

# (optional) README-Ergebnistabellen (6.1 & 6.3) aus results/*.csv befüllen
python .github/_update_readme_results.py
```

### Automatische Aktualisierung via GitHub Actions

Der Workflow [`update-metrics.yml`](.github/workflows/update-metrics.yml) führt
das Notebook auf einem GitHub-Actions-Runner (mit Internetzugang zu Yahoo
Finance) vollständig aus, befüllt die README-Ergebnistabellen und öffnet
automatisch einen Pull Request zur Prüfung. Manuell auslösbar über den
Tab **Actions → Update metrics (notebook run) → Run workflow**.

Zusätzlich prüft [`sanity-check.yml`](.github/workflows/sanity-check.yml)
automatisch bei jedem Push/PR auf `main`, ob die `.github/*.py`-Skripte
syntaktisch korrekt sind, das Notebook gültiges JSON ist und die
Workflow-YAMLs valide sind — ohne Netzwerkzugriff oder schwere
Abhängigkeiten (TensorFlow etc.).

### Reproduzierbarkeitsgarantien

| Massnahme | Detail |
|-----------|--------|
| Globaler Seed | `SEED = 42` (NumPy, Python random, TensorFlow) |
| Chronologischer Split | Kein zeitliches Datenleck möglich |
| Scaler-Kalibrierung | Ausschliesslich auf Trainingsdaten |
| Konfiguration | Alle Parameter zentral in Zelle 0.1 |
| Offlinefähigkeit | CSV-Snapshot- und synthetischer GBM-Fallback bei fehlendem Netzwerk |

### Offline-Betrieb / Datenquellen-Fallback

`load_price_data()` versucht die Datenbeschaffung in drei Stufen, jeweils nur
falls die vorherige Stufe fehlschlägt:

1. **Live-Download via `yfinance`** (Yahoo Finance) — Standardfall mit
   Netzwerkzugriff.
2. **CSV-Snapshot** aus [`data/nvda_ohlcv.csv`](data/nvda_ohlcv.csv), falls
   vorhanden — ein einmalig über den Workflow
   [`fetch-data.yml`](.github/workflows/fetch-data.yml) heruntergeladener,
   echter historischer Datensatz (siehe unten). Ermöglicht reproduzierbare
   Ausführung mit **echten** Kursdaten auch ganz ohne Netzwerkzugriff.
3. **Synthetischer, NVIDIA-ähnlicher Ersatzdatensatz** (Geometrische
   Brownsche Bewegung), falls auch kein Snapshot vorliegt:

```
dS_t = mu * S_t * dt + sigma * S_t * dW_t

mit: mu    = 0.0008  (tägliche Drift)
     sigma = 0.025   (tägliche Volatilität)
```

Die Variable `DATA_SOURCE` dokumentiert die tatsächlich verwendete Quelle
(`"yfinance"` · `"csv_snapshot"` · `"synthetic"`).

> **Für belastbare wissenschaftliche Aussagen** sind stets echte Kursdaten zu
> verwenden (`DATA_SOURCE in {"yfinance", "csv_snapshot"}`), nicht der rein
> illustrative synthetische Fallback.

### Kursdaten-Snapshot aktualisieren

Der Workflow [`fetch-data.yml`](.github/workflows/fetch-data.yml) lädt den
aktuellen Kursverlauf (Ticker/Zeitraum aus dem Notebook, Zelle 0.1)
via `yfinance` auf einem GitHub-Actions-Runner (mit Internetzugang) herunter
und öffnet einen Pull Request mit dem aktualisierten
[`data/nvda_ohlcv.csv`](data/nvda_ohlcv.csv). Manuell auslösbar über
**Actions → Fetch NVDA price data → Run workflow**.

### Interaktive Web-App (Streamlit)

Ergänzend zum Notebook steht unter [`app/`](app/) eine **Streamlit-App** mit
zwei Ansichten bereit: ein **Dashboard**, das die `results/*.csv`-Ausgaben
eines Notebook-Laufs visualisiert, sowie eine **Live-Prognose**, die aktuelle
Kursdaten lädt und in Echtzeit eine Naive-/ARIMA-Prognose erstellt (das LSTM
wird dort bewusst nicht verwendet — siehe [`app/README.md`](app/README.md)
für die Begründung und Startanleitung).

```bash
pip install -r requirements.txt -r app/requirements.txt
streamlit run app/Home.py
```

---

## 9  Projektstruktur

```
stock-price-prediction/
├── NVIDIA_Kursprognose_LSTM.ipynb   # Hauptabgabe — vollständiges Notebook
├── requirements.txt                  # Python-Abhängigkeiten (Kern)
├── .github/
│   ├── workflows/                    # CI: Notebook-Ausführung, Daten-Snapshot, Sanity-Check bei jedem PR
│   ├── _fetch_data.py                # Lädt CSV-Snapshot (data/nvda_ohlcv.csv)
│   └── _update_readme_results.py     # Befüllt README-Tabellen (6.1, 6.3) aus results/*.csv
├── app/                               # Streamlit-Web-App (Dashboard + Live-Prognose)
│   ├── Home.py                       # Einstiegsseite
│   ├── common.py                     # Gemeinsame Data-Loading-/Feature-Logik
│   ├── pages/                        # Dashboard- und Live-Prognose-Seiten
│   └── requirements.txt              # Zusätzliche Abhängigkeiten (streamlit, plotly)
├── TODO.md                           # Offene To-Dos & Projektplan
└── README.md                         # Diese Datei
```

---

## 10  Abhängigkeiten

### Kern (zwingend)

| Paket | Version | Verwendung |
|-------|---------|-----------|
| `numpy` | >= 1.26 | Numerik, Array-Operationen |
| `pandas` | >= 2.0 | Datenmanipulation, Zeitreihen |
| `scipy` | >= 1.11 | Statistische Tests |
| `matplotlib` | >= 3.8 | Visualisierungen |
| `seaborn` | >= 0.13 | Statistische Grafiken |
| `scikit-learn` | >= 1.4 | Scaler, Metriken, MLP-Fallback |
| `statsmodels` | >= 0.14 | ADF/KPSS-Tests, ACF/PACF, ARIMA/SARIMA |
| `yfinance` | >= 0.2.40 | NVIDIA-Kursdaten (Yahoo Finance) |
| `tqdm` | >= 4.60 | Fortschrittsbalken (ARIMA-Suche, LSTM-HPO) |

### Optional (empfohlen)

| Paket | Version | Verwendung | Fallback |
|-------|---------|-----------|---------|
| `tensorflow-cpu` | >= 2.16 | LSTM-Training | MLP via `scikit-learn` |
| `shap` | >= 0.45 | Feature-Explainability | Permutationswichtigkeit |
| `plotly` | >= 5.20 | Interaktive Visualisierung (Kap. 7.2d) | Zelle wird übersprungen |

---

## Literatur

- Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation, 9(8), 1735–1780.
- Hyndman, R. J. & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.
- Lundberg, S. M. & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS.
- Fama, E. F. (1970). *Efficient Capital Markets: A Review of Theory and Empirical Work.* Journal of Finance, 25(2), 383–417.
- Box, G. E. P., Jenkins, G. M., Reinsel, G. C. & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.

---

*Alle Visualisierungen, Metriken und Modellgewichte werden bei der Ausführung
des Notebooks generiert. Sie sind nicht Teil des Repository.*

