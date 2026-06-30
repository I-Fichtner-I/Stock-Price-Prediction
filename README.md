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
11. [To-Do & Projektplan](#11-to-do--projektplan)

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
│   ├── Naive Baseline              ├── Architektur-Suche         │
│   ├── Moving-Average-Baseline     │   (Schichten, Neuronen,     │
│   └── ARIMA/SARIMA (AIC-Suche)   │    Dropout, Lernrate)        │
│                                   └── Walk-Forward-CV           │
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
Zeitreihen-CV mit expandierendem Trainingsfenster (5 Folds):

Fold 1:  [████████████████][░░░░░░░░░░░░░░░░░░░░░░░░]
Fold 2:  [████████████████████][░░░░░░░░░░░░░░░░░░░░]
Fold 3:  [████████████████████████][░░░░░░░░░░░░░░░░]
Fold 4:  [████████████████████████████][░░░░░░░░░░░░]
Fold 5:  [████████████████████████████████][░░░░░░░░]

         [████] Training   [░░░░] Test (je Fold)
```

---

## 6  Zentrale Ergebnisse

> Die nachfolgenden Werte sind **laufzeitabhängig** (echte Yahoo-Finance-Daten
> vs. synthetischer Fallback, gewählte Hyperparameter). Alle Zahlen werden bei
> Ausführung des Notebooks reproduzierbar in `results/` als CSV gespeichert.

### 6.1  Modellvergleich (Testmenge)

```
Modell               │ MAE (USD) │ RMSE (USD) │ MAPE (%) │ Theil's U │ DA (%)
─────────────────────┼───────────┼────────────┼──────────┼───────────┼──────────
Naive Baseline       │     –     │      –     │    –     │   1.000   │  ~50.0
Moving Average       │     –     │      –     │    –     │   > 1.0   │    –
ARIMA/SARIMA         │     –     │      –     │    –     │   ~1.0    │    –
LSTM (Niveau)        │     –     │      –     │    –     │   ~1.0    │    –
LSTM (Log-Renditen)  │     –     │      –     │    –     │    –      │    –
─────────────────────┴───────────┴────────────┴──────────┴───────────┴──────────
Werte werden bei Notebook-Ausführung befüllt (→ results/metrics_comparison.csv)
```

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

| Zeitreihe | ADF p-Wert | KPSS-Befund | Schluss |
|-----------|-----------|-------------|---------|
| Close (Niveau) | > 0.05 | stationär abgelehnt | **nicht stationär** |
| Log-Renditen | < 0.01 | stationär | **stationär** |
| Einfache Renditen | < 0.01 | stationär | stationär |

*(Genaue Testwerte werden bei Notebook-Ausführung eingetragen)*

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

# (optional) Notebook aus dem Generator-Skript neu erzeugen
python _build_notebook.py
```

### Reproduzierbarkeitsgarantien

| Massnahme | Detail |
|-----------|--------|
| Globaler Seed | `SEED = 42` (NumPy, Python random, TensorFlow) |
| Chronologischer Split | Kein zeitliches Datenleck möglich |
| Scaler-Kalibrierung | Ausschliesslich auf Trainingsdaten |
| Konfiguration | Alle Parameter zentral in Zelle 0.1 |
| Offlinefähigkeit | Synthetischer GBM-Fallback bei fehlendem Netzwerk |

### Offline-Betrieb / Synthetische Daten

Ist kein Netzwerkzugriff verfügbar, erzeugt das Notebook automatisch einen
**synthetischen, NVIDIA-ähnlichen Datensatz** (Geometrische Brownsche Bewegung):

```
dS_t = mu * S_t * dt + sigma * S_t * dW_t

mit: mu    = 0.0008  (tägliche Drift)
     sigma = 0.025   (tägliche Volatilität)
```

Die Variable `DATA_SOURCE` dokumentiert die tatsächlich verwendete Quelle.

> **Für belastbare wissenschaftliche Aussagen** sind stets die echten
> Yahoo-Finance-Daten zu verwenden (`DATA_SOURCE == "yfinance"`).

---

## 9  Projektstruktur

```
stock-price-prediction/
├── NVIDIA_Kursprognose_LSTM.ipynb   # Hauptabgabe — vollständiges Notebook
├── requirements.txt                  # Python-Abhängigkeiten (Kern)
├── _build_notebook.py                # Generator-Skript (reproduzierbare Zellstruktur)
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

---

## 11  To-Do & Projektplan

Dieser Abschnitt dokumentiert offene Verbesserungspotenziale, geordnet nach
**Priorität** und **Aufwand**. Die Kategorien orientieren sich am
wissenschaftlichen Erkenntnisgewinn: Daten → Modelle → Evaluation → Engineering.

### Legende

```
Priorität:  [H] Hoch · [M] Mittel · [N] Niedrig
Aufwand:    [S] Klein (<1 Tag) · [M] Mittel (2–5 Tage) · [L] Gross (>1 Woche)
Status:     [ ] Offen · [~] In Arbeit · [x] Erledigt
```

---

### Phase 1 — Wissenschaftliche Robustheit  *(kurzfristig)*

```
[ ] [H][S]  Metriktabelle mit echten Zahlenwerten befüllen
            --> Notebook einmal vollständig ausführen, results/*.csv einlesen,
                Tabelle in Abschnitt 6.1 automatisch befüllen (nbconvert + Jinja).

[ ] [H][M]  Statistische Signifikanztests ergänzen
            --> Diebold-Mariano-Test: prüft, ob LSTM-Fehler signifikant kleiner
                als Baseline-Fehler (H0: gleiche Prognosegüte).
            --> Wilcoxon-Vorzeichen-Rang-Test auf Residuen.
            --> Implementierung: statsmodels.stats.diagnostic.acorr_ljungbox
                bereits vorhanden; dm_test als eigenständige Funktion ergänzen.

[ ] [H][S]  Konfidenzintervalle / Prognoseintervalle ausgeben
            --> Bootstrap-Methode auf LSTM-Residuen (1 000 Resamplings).
            --> Ziel: Visualisierung von 80 %- und 95 %-Bändern im Ist/Soll-Plot.

[ ] [M][S]  KPSS-Testwerte in Stationaritätstabelle (Abschnitt 6.3) eintragen
            --> Automatisch aus Notebook-Output extrahieren.

[ ] [M][M]  Residualdiagnose für LSTM ergänzen
            --> ACF/PACF der Testresiduen, Ljung-Box-Test auf Autokorrelation.
            --> Quantil-Quantil-Plot (QQ-Plot) für Normalverteilungsannahme.
```

---

### Phase 2 — Modellbreite  *(mittelfristig)*

```
[ ] [H][M]  Transformer / Temporal Fusion Transformer (TFT) hinzufügen
            --> Attention-Mechanismus als Alternative zu LSTM-Gating.
            --> Paket: pytorch-forecasting oder neuralforecast (Nixtla).
            --> Erlaubt direkte Quantilsprognosen ohne Bootstrap.

[ ] [H][M]  Gradient Boosting (XGBoost / LightGBM) als weiteres Referenzmodell
            --> Kein Sequence-Modell, aber starker tabellarischer Baseline.
            --> Direkt auf Feature-Matrix (ohne Sliding Window) anwendbar.
            --> Vergleich: Entscheidet, ob der Sequence-Anteil des LSTM hilft.

[ ] [M][L]  GRU (Gated Recurrent Unit) als LSTM-Variante
            --> Geringere Parameterzahl, oft vergleichbare Genauigkeit.
            --> Gleicher Hyperparameter-Suchraum wie LSTM -- fairer Vergleich.

[ ] [M][M]  Multi-Step-Forecasting (Horizont h = 5, 10, 20 Handelstage)
            --> Direkte vs. rekursive Mehrschritt-Strategie vergleichen.
            --> Separate Fehlermetriken je Horizont (MAE@h, RMSE@h).

[ ] [N][S]  Prophet (Meta) als saisonales Baseline-Modell
            --> Robustes additives Modell mit Trendbrüchen und Feiertagen.
            --> Schnell implementiert, gute Interpretierbarkeit.
```

---

### Phase 3 — Datenerweiterung  *(mittelfristig)*

```
[ ] [H][M]  Sentiment-Features aus Finanznachrichten integrieren
            --> Quellen: NewsAPI, VADER (lexikonbasiert) oder FinBERT (transformer).
            --> Täglicher Sentiment-Score als zusätzliches Feature.
            --> Hypothese: Nachrichtensentiment verbessert Richtungsgenauigkeit (DA).

[ ] [H][M]  Makroökonomische Kontextvariablen ergänzen
            --> VIX (Volatilitätsindex), Fed Funds Rate, USD/EUR-Kurs, SOX-Index.
            --> Alle tagesaktuell via yfinance oder FRED (fredapi) verfügbar.

[ ] [M][M]  Multivariate Zeitreihe: weitere Halbleiteraktien einbeziehen
            --> AMD, Intel, TSMC als korrelierte Zeitreihen.
            --> Erlaubt Cross-Asset-Signale im LSTM (vektorieller Input).

[ ] [M][L]  Orderbook-/Intraday-Daten (15-Min-Intervalle)
            --> Feinere Zeitauflösung für kurzfristige Prognosen.
            --> Höheres Rauschen, aber potentiell stärkere Autokorrelationsstruktur.

[ ] [N][S]  Erweiterung des Beobachtungszeitraums (ab 2015)
            --> Schliesst weitere Marktregimes ein (GPU-Boom 2016/17, COVID-Crash).
```

---

### Phase 4 — Evaluation & Backtesting  *(mittelfristig)*

```
[ ] [H][M]  Realistische Trading-Simulation ausbauen
            --> Transaktionskosten (Spread + Kommission ca. 0.1 %) berücksichtigen.
            --> Slippage-Modell für grosse Positionen.
            --> Risikokennzahlen: Sharpe Ratio, Maximum Drawdown, Calmar Ratio.

[ ] [H][M]  Out-of-Sample-Test auf anderer Aktie (Generalisierbarkeit)
            --> Gleiche Pipeline auf AMD oder Tesla anwenden.
            --> Prüft, ob Ergebnisse ticker-spezifisch oder allgemein gültig sind.

[ ] [M][M]  Regime-aware Evaluation
            --> Testmenge nach Marktregime aufteilen (Trend, Seitwärts, Crash).
            --> Separate Metriken je Regime: Wo versagen die Modelle?

[ ] [M][S]  Benchmark gegen Random-Walk-Simulation
            --> Monte-Carlo-Simulation (1 000 Pfade) als statistischer Untergrenze.
            --> Vergleich: Liegt LSTM-RMSE unter dem Erwartungswert des Random Walk?
```

---

### Phase 5 — Code-Qualität & Engineering  *(laufend)*

```
[ ] [H][M]  Notebook in modulare Python-Pakete refaktorieren
            --> src/data/loader.py, src/features/engineering.py,
                src/models/lstm.py, src/evaluation/metrics.py
            --> Notebook wird zum reinen Präsentations-Layer (ruft Module auf).

[ ] [H][S]  Automatische Metriken-Befüllung im README via CI
            --> GitHub Action: führt Notebook headless aus, extrahiert
                results/metrics_comparison.csv, aktualisiert README-Tabelle.

[ ] [M][M]  Unit-Tests für Kernfunktionen
            --> pytest: Feature-Engineering, Sliding-Window, Metrikberechnungen.
            --> Verhindert stille Regressionen bei Refaktorierungen.

[ ] [M][M]  Interaktive Visualisierungen (Plotly / Dash)
            --> Zoom-fähige Kursverläufe, interaktiver Modellvergleich.
            --> Exportierbar als HTML (kein Server nötig).

[ ] [N][L]  MLflow / Weights & Biases für Experiment-Tracking
            --> Automatisches Logging von Hyperparametern, Metriken, Artefakten.
            --> Vergleich vieler Runs ohne manuelle Tabellenpflege.

[ ] [N][M]  Docker-Container für vollständige Reproduzierbarkeit
            --> Dockerfile mit pinned Versionen (requirements.txt + system libs).
            --> Notebook läuft identisch auf jeder Maschine.
```

---

### Priorisierungsmatrix

```
              AUFWAND
              Klein (S)       Mittel (M)       Gross (L)
            ┌────────────────┬────────────────┬────────────────┐
  HOCH  [H] │ Metriktabelle  │ Signifikanztests│ TFT/Transformer│
            │ KPSS-Werte     │ XGBoost Baseline│                │
            │ Konfidenzband  │ Sentiment-Feat. │                │
            │                │ Trading-Sim.    │                │
            │                │ Modularisierung │                │
            ├────────────────┼────────────────┼────────────────┤
  MITTEL [M]│ Prophet        │ GRU-Vergleich  │ GRU (gross)    │
            │ Random-Walk-   │ Multi-Step-FC  │ Intraday-Daten │
            │ Benchmark      │ Makrodaten     │                │
            │                │ Regime-Eval.   │                │
            │                │ Unit-Tests     │                │
            │                │ Plotly-Viz.    │                │
            ├────────────────┼────────────────┼────────────────┤
  NIEDRIG[N]│ Zeitraum-Ext.  │ Docker         │ MLflow         │
            │                │                │                │
            └────────────────┴────────────────┴────────────────┘

  Empfohlene Startreihenfolge:
  1. Metriktabelle befüllen (H/S) -- sofortiger Mehrwert, kein Aufwand
  2. Konfidenzintervalle (H/S)    -- stärkt wissenschaftliche Aussagekraft
  3. Signifikanztests (H/M)       -- ohne diese sind Vergleiche angreifbar
  4. XGBoost Baseline (H/M)       -- klärt, ob Sequence-Modell überhaupt hilft
  5. Sentiment-Features (H/M)     -- grösste potenzielle DA-Verbesserung
```

---

### Offene Forschungsfragen

| # | Frage | Adressiert durch |
|---|-------|-----------------|
| F1 | Verbessert Sentiment-Information die Richtungsgenauigkeit messbar (DA > 55 %)? | Phase 3: Sentiment-Features |
| F2 | Generalisiert die Pipeline auf andere Ticker ohne Neutraining? | Phase 4: Out-of-Sample-Test |
| F3 | Schlägt ein Transformer das LSTM auf dieser Datenbasis? | Phase 2: TFT |
| F4 | Ist der Mehrwert des LSTM gegenüber XGBoost durch den Sequence-Anteil bedingt? | Phase 2: XGBoost |
| F5 | Wie verhalten sich die Modelle in unterschiedlichen Marktregimes (Crash vs. Boom)? | Phase 4: Regime-Eval |
