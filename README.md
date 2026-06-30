# NVIDIA-Kursprognose mit LSTM — Begleit-Notebook

Dieses Verzeichnis enthält das Jupyter-Notebook zur Seminar-/Termpaper-Arbeit
**„Prognose von Aktienkursen mittels maschinellen Lernens am Beispiel der
NVIDIA-Aktie"**. Es setzt den im Exposé beschriebenen Untersuchungsablauf
vollständig und reproduzierbar um — mit ausführlichen Erklärungen in Markdown
(Deutsch), lauffähigem Python-Code und Visualisierungen.

## Dateien

| Datei | Beschreibung |
|---|---|
| `NVIDIA_Kursprognose_LSTM.ipynb` | **Hauptabgabe** — das vollständige, bereits ausgeführte Notebook. |
| `requirements.txt` | Python-Abhängigkeiten für das Notebook. |
| `_build_notebook.py` | Generator-Skript, das das `.ipynb` reproduzierbar erzeugt (Dokumentation der Zellstruktur). |

## Aufbau des Notebooks

Die Gliederung folgt dem Exposé:

0. **Setup & Reproduzierbarkeit** — Bibliotheken, Seeds, optionale Pakete.
3. **Datengrundlage** — Bezug via Yahoo Finance (`yfinance`) inkl. Datenqualitäts-
   prüfung und Data Cleaning.
4. **Explorative Datenanalyse** — Deskriptive Statistik, Rendite-/Preisanalysen,
   zeitliche Strukturen, **ADF-Stationaritätstest**, Autokorrelation (ACF/PACF),
   Volumen-/Liquiditäts- und Korrelationsanalysen, Ereignisanalyse.
5. **Datenvorverarbeitung & Feature Engineering** — abgeleitete Merkmale
   (Renditen, gleitende Durchschnitte, Volatilität, Lag/Volumen, Wochentag),
   Skalierung, **Sliding-Window-Transformation**, strikt chronologischer
   Train/Val/Test-Split.
6. **Modellierung** — Referenzmodelle (naiv, Moving Average), ein
   **ARIMA/SARIMA mit automatischer Ordnungswahl** (AIC-Gittersuche, Walk-Forward-
   Prognose) sowie ein **LSTM mit eigenständiger Hyperparameter-Suche**
   (Variation von Schichten, Neuronen und Dropout; Auswahl nach Validierungs-RMSE).
7. **Evaluation & Explainability** — MAE/MSE/RMSE, Vergleich aller Modellfamilien,
   Ist-/Prognosevergleich, Overfitting-Prüfung, **SHAP**-Analyse (Merkmals- und
   Zeitschritt-Wichtigkeit).
8. **Diskussion** — kritische Einordnung inkl. Lag-Diagnose und Richtungsgenauigkeit.
9. **Fazit & Ausblick**.

## Ausführung

```bash
# 1) Abhängigkeiten installieren
pip install -r notebooks/requirements.txt

# 2a) Interaktiv im Browser
jupyter notebook notebooks/NVIDIA_Kursprognose_LSTM.ipynb

# 2b) Oder headless ausführen (alle Zellen neu rechnen)
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1200 \
  notebooks/NVIDIA_Kursprognose_LSTM.ipynb

# (optional) Notebook aus dem Generator-Skript neu erzeugen
python notebooks/_build_notebook.py
```

## Hinweise

- **Datenquelle / Offline-Betrieb.** Das Notebook lädt echte NVIDIA-Kursdaten
  über Yahoo Finance. Ist kein Netzwerkzugriff möglich (z. B. in einer
  abgeschotteten Umgebung), erzeugt es automatisch einen **synthetischen,
  NVIDIA-ähnlichen Ersatzdatensatz**, damit es jederzeit vollständig durchläuft.
  Die tatsächlich genutzte Quelle wird in der Variable `DATA_SOURCE`
  dokumentiert. **Für belastbare wissenschaftliche Aussagen sind stets die echten
  Yahoo-Finance-Daten zu verwenden** (`DATA_SOURCE == "yfinance"`).
- **Selbstoptimierung der Modelle.** ARIMA/SARIMA wählt seine Ordnung
  $(p,d,q)(P,D,Q)_m$ automatisch über eine **zweistufige AIC-Suche** (Stufe 1:
  großes nicht-saisonales Gitter $p\in\{0..4\}$, $d\in\{1,2\}$, $q\in\{0..4\}$;
  Stufe 2: saisonale Verfeinerung mit $m=5$). Das LSTM durchsucht einen
  **erweiterten Konfigurationsraum** (1–3 Schichten, 32–128 Neuronen, Dropout,
  Lernrate) und übernimmt die beste Variante nach Validierungs-RMSE. Der Suchraum
  lässt sich in den jeweiligen Zellen (Variablen `p_range`/`d_range`/`q_range`,
  `seasonal_candidates`, `search_space`) einfach weiter anpassen.
- **Optionale Pakete.** Fehlt **TensorFlow**, nutzt das Notebook ein
  MLP-Ersatzmodell (`scikit-learn`); fehlt **SHAP**, kommt eine
  modell-agnostische Permutationswichtigkeit zum Einsatz. Die Auswertung bleibt
  in beiden Fällen identisch strukturiert. `statsmodels` (für ARIMA/SARIMA) ist
  Bestandteil der Kernabhängigkeiten.
- **Reproduzierbarkeit.** Alle Zufalls-Seeds sind fixiert (`SEED = 42`); die
  Train/Val/Test-Aufteilung erfolgt streng chronologisch, und der Scaler wird nur
  auf den Trainingsdaten kalibriert (Vermeidung von Look-Ahead-Bias).
