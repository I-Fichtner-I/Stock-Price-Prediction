# To-Do & Projektplan

Dieser Abschnitt dokumentiert offene Verbesserungspotenziale, geordnet nach
**Priorität** und **Aufwand**. Die Kategorien orientieren sich am
wissenschaftlichen Erkenntnisgewinn: Daten → Modelle → Evaluation → Engineering.

> Ausgelagert aus der README (Abschnitt "To-Do & Projektplan"); enthält nur
> noch offene (`[ ]`) bzw. in Arbeit befindliche (`[~]`) Punkte. Erledigte
> Punkte wurden entfernt.

### Legende

```
Priorität:  [H] Hoch · [M] Mittel · [N] Niedrig
Aufwand:    [S] Klein (<1 Tag) · [M] Mittel (2–5 Tage) · [L] Gross (>1 Woche)
Status:     [ ] Offen · [~] In Arbeit
```

---

### Phase 1 — Wissenschaftliche Robustheit  *(kurzfristig)*

```
[~] [H][S]  Metriktabelle mit echten Zahlenwerten befüllen
            --> Automatisierung steht (_update_readme_results.py liest results/*.csv
                und befüllt Abschnitt 6.1 zwischen AUTO-GENERATED-Markern).
            --> Offen: einmal mit echten Yahoo-Finance-Daten ausführen und README
                committen (in dieser Sandbox kein Netzwerkzugriff auf Yahoo Finance).

[~] [M][S]  KPSS-Testwerte in Stationaritätstabelle (Abschnitt 6.3) eintragen
            --> Automatisch aus Notebook-Output extrahieren.
            --> Export (stationarity_tests.csv) und Befüll-Skript stehen; Tabelle
                selbst wird erst mit echten Daten final committet (s.o.).
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
```

---

### Phase 5 — Code-Qualität & Engineering  *(laufend)*

```
[ ] [H][M]  Notebook in modulare Python-Pakete refaktorieren
            --> src/data/loader.py, src/features/engineering.py,
                src/models/lstm.py, src/evaluation/metrics.py
            --> Notebook wird zum reinen Präsentations-Layer (ruft Module auf).

[ ] [M][M]  Unit-Tests für Kernfunktionen
            --> pytest: Feature-Engineering, Sliding-Window, Metrikberechnungen.
            --> Verhindert stille Regressionen bei Refaktorierungen.

[ ] [N][L]  MLflow / Weights & Biases für Experiment-Tracking
            --> Automatisches Logging von Hyperparametern, Metriken, Artefakten.
            --> Vergleich vieler Runs ohne manuelle Tabellenpflege.
```

---

### Priorisierungsmatrix

```
              AUFWAND
              Klein (S)       Mittel (M)       Gross (L)
            ┌────────────────┬────────────────┬────────────────┐
  HOCH  [H] │ Metriktabelle~ │ XGBoost Baseline│ TFT/Transformer│
            │ KPSS-Werte~    │ Sentiment-Feat. │                │
            │                │ Trading-Sim.    │                │
            │                │ Modularisierung │                │
            ├────────────────┼────────────────┼────────────────┤
  MITTEL [M]│ –              │ GRU-Vergleich  │ GRU (gross)    │
            │                │ Multi-Step-FC  │ Intraday-Daten │
            │                │ Makrodaten     │                │
            │                │ Regime-Eval.   │                │
            │                │ Unit-Tests     │                │
            ├────────────────┼────────────────┼────────────────┤
  NIEDRIG[N]│ –              │ –              │ MLflow         │
            └────────────────┴────────────────┴────────────────┘

  ~ = [~] in Arbeit: Automatisierung steht, wartet auf einen Notebook-Lauf mit
      echten Marktdaten (z.B. via .github/workflows/update-metrics.yml).

  Empfohlene Startreihenfolge:
  1. XGBoost Baseline (H/M)            -- klärt, ob Sequence-Modell überhaupt hilft
  2. Sentiment-Features (H/M)          -- grösste potenzielle DA-Verbesserung
  3. Modularisierung (H/M)             -- Voraussetzung für Unit-Tests (Phase 5)
  4. Trading-Simulation ausbauen (H/M) -- realistischere ökonomische Bewertung
  5. Transformer/TFT (H/L)             -- grösster Umbau, höchster Erkenntnisgewinn
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
