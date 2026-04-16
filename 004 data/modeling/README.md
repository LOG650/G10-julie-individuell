# Modellering

Denne mappen inneholder første versjon av modelleringspipelinen for rapporten.

Målet er å sammenligne fire modeller mot samme problemstilling:

- `SARIMA`
- `Eksponentiell glatting`
- `XGBoost`
- `LSTM`

## Datagrunnlag

Pipelinen bruker `004 data/Data som skal brukes Anonymisert.csv` som masterfil.
Fra denne genereres to eksplisitte råformat-filer i `004 data/`:

- `train.csv`
- `test.csv`

Disse brukes til historisk evaluering av modellene med en fast tidsbasert splitt:

- `train`: april 2021 til desember 2024
- `test`: januar 2025 til mars 2026

Det opprinnelige datasettet dekker månedlige observasjoner fra april 2021 til mars 2026.
Det gir normalt inntil 60 tidssteg per fartøy, men med to praktiske begrensninger:

- 2021 starter i april fordi dataserien dekker de siste fem årene
- 2026 er foreløpig bare observert til og med mars

Standardkjøringen dekker nå to faser i samme pipeline:

- historisk modellbygging, testing og verifikasjon
- fremtidsprognoser på full historikk

## Oppsett

For å unngå støy i Git og VS Code bør det virtuelle miljøet ligge utenfor repoet.
På denne maskinen brukes den permanente stien
`/Users/juliekarlsen/.venvs/g10-julie-individuell`.

Hvis miljøet allerede finnes, aktiver det slik:

```bash
source /Users/juliekarlsen/.venvs/g10-julie-individuell/bin/activate
```

Hvis miljøet må opprettes på nytt, bruk:

```bash
python3 -m venv /Users/juliekarlsen/.venvs/g10-julie-individuell
source /Users/juliekarlsen/.venvs/g10-julie-individuell/bin/activate
pip install -r '004 data/modeling/requirements.txt'
```

## Kjøring

Generer først eksplisitte train/test-filer:

```bash
python '004 data/modeling/generate_train_test_split.py'
```

Standardskriptet kjører nå alle fire modellene:

- `SARIMA`
- `Eksponentiell glatting`
- `XGBoost`
- `LSTM`

```bash
python '004 data/modeling/run_models.py'
```

Skriptet skriver testresultater, sammendrag, fremtidsprognoser og figurer til `004 data/modeling/results/`.
I tillegg skriver det modellspesifikke metode-, resultat- og diagnosefiler til egne mapper under `004 data/modeling/models/`.
Ved hver kjøring verifiserer og oppdaterer skriptet også `train.csv` og `test.csv` fra masterfilen.

Fremtidsprognosene bygges på hele datasettet til og med `2026-03` og lagres for fire horisonter:

- `1` måned fram
- `3` måneder fram
- `6` måneder fram
- `12` måneder fram

## Versjonering

For leveransen skal følgende under `004 data/` ligge i Git:

- den anonymiserte CSV-filen som brukes som datagrunnlag
- `train.csv` og `test.csv` som eksplisitt train/test-splitt for modellene
- modelleringskode og avhengigheter i `004 data/modeling/`
- genererte resultater i `004 data/modeling/results/`
- modellspesifikke kode- og resultatfiler i `004 data/modeling/models/<modell>/`
- en samlet sammenligningsfil i `004 data/modeling/model_logs/Modellsammenligning.md`

Dette gjør at både datagrunnlag, kode og siste kjørte modellresultater kan spores og leveres samlet.

## Utdata

Skriptet lager som standard:

- `004 data/modeling/results/metrics.json`
- `004 data/modeling/results/predictions.csv`
- `004 data/modeling/results/model_comparison_summary.csv`
- `004 data/modeling/results/model_comparison_summary.md`
- `004 data/modeling/results/metrics_by_vessel.csv`
- `004 data/modeling/results/metrics_by_vessel.md`
- `004 data/modeling/results/metrics_by_month.csv`
- `004 data/modeling/results/metrics_by_month.md`
- `004 data/modeling/results/future_predictions.csv`
- `004 data/modeling/results/future_totals_by_model_and_date.csv`
- `004 data/modeling/results/future_predictions_1m.csv`
- `004 data/modeling/results/future_predictions_1m_pivot.csv`
- `004 data/modeling/results/future_predictions_3m.csv`
- `004 data/modeling/results/future_predictions_3m_pivot.csv`
- `004 data/modeling/results/future_predictions_6m.csv`
- `004 data/modeling/results/future_predictions_6m_pivot.csv`
- `004 data/modeling/results/future_predictions_12m.csv`
- `004 data/modeling/results/future_predictions_12m_pivot.csv`
- `004 data/modeling/results/figures/mae_per_model.png`
- `004 data/modeling/results/figures/mae_by_month.png`
- `004 data/modeling/results/figures/mae_heatmap_by_vessel.png`
- `004 data/modeling/results/figures/future_total_offhire_1m.png`
- `004 data/modeling/results/figures/future_total_offhire_3m.png`
- `004 data/modeling/results/figures/future_total_offhire_6m.png`
- `004 data/modeling/results/figures/future_total_offhire_12m.png`
- `004 data/modeling/models/<modell>/metrics.json`
- `004 data/modeling/models/<modell>/predictions.csv`
- `004 data/modeling/models/<modell>/future_predictions.csv`
- `004 data/modeling/models/<modell>/metode.md`
- `004 data/modeling/models/<modell>/kode.md`
- `004 data/modeling/models/<modell>/resultater.md`
- `004 data/modeling/model_logs/Modellsammenligning.md`

For `SARIMA` genereres det i tillegg Box-Jenkins-artefakter og diagnostikk:

- `004 data/modeling/models/SARIMA/stasjonaritet.csv`
- `004 data/modeling/models/SARIMA/stasjonaritet.md`
- `004 data/modeling/models/SARIMA/kandidatmodeller.csv`
- `004 data/modeling/models/SARIMA/kandidatmodeller.md`
- `004 data/modeling/models/SARIMA/modellvalg_per_fartoy.csv`
- `004 data/modeling/models/SARIMA/modellvalg_per_fartoy.md`
- `004 data/modeling/models/SARIMA/residualdiagnostikk.csv`
- `004 data/modeling/models/SARIMA/residualdiagnostikk.md`
- `004 data/modeling/models/SARIMA/acf.png`
- `004 data/modeling/models/SARIMA/pacf.png`
- `004 data/modeling/models/SARIMA/residualdiagnostikk.png`
- `004 data/modeling/models/SARIMA/representativ_testplot.png`

For `Eksponentiell glatting` genereres det i tillegg:

- `004 data/modeling/models/Eksponentiell glatting/modellvalg_per_fartoy.csv`
- `004 data/modeling/models/Eksponentiell glatting/modellvalg_per_fartoy.md`
- `004 data/modeling/models/Eksponentiell glatting/residualdiagnostikk.csv`
- `004 data/modeling/models/Eksponentiell glatting/residualdiagnostikk.md`
- `004 data/modeling/models/Eksponentiell glatting/representativ_testplot.png`

For `XGBoost` genereres det i tillegg:

- `004 data/modeling/models/XGBoost/feature_importance.csv`
- `004 data/modeling/models/XGBoost/feature_importance.md`
- `004 data/modeling/models/XGBoost/feature_importance.png`
- `004 data/modeling/models/XGBoost/representativ_testplot.png`

For `LSTM` genereres det i tillegg:

- `004 data/modeling/models/LSTM/training_history.csv`
- `004 data/modeling/models/LSTM/training_history.md`
- `004 data/modeling/models/LSTM/training_history.png`
- `004 data/modeling/models/LSTM/representativ_testplot.png`

I tillegg skriver det en kort oppsummering til terminalen om hvilke modeller som ble kjørt, og hvilke som eventuelt ble hoppet over på grunn av datamengde.

## Evalueringslogikk

Alle fire modellene evalueres nå under samme hovedoppsett:

- målnivå: månedlig `offhire_days` per fartøy
- train: `2021-04` til `2024-12`
- test: `2025-01` til `2026-03`
- metode: ekspanderende `1-stegs` prognose gjennom testperioden
- hovedmål: `MAE`
- støttemål: `RMSE` og `sMAPE`

Dette gjør at modellene kan sammenlignes direkte på samme fartøy-måned-observasjoner.

## Fremtidsprognoser

Når historisk evaluering er gjennomført, brukes hele datasettet til og med `2026-03` til å bygge fremtidsprognoser.
Disse evalueres ikke med `MAE`, `RMSE` eller `sMAPE`, siden faktiske observasjoner ikke finnes ennå.

For hver modell lagres først en basisprognose på `12` steg fram per fartøy i modellmappen.
Deretter skrives egne samlede artefakter for `1`, `3`, `6` og `12` måneder fram slik at resultatene kan brukes direkte i rapporten.

Felles kolonner i de samlede forecast-filene under `004 data/modeling/results/` er:

- `model`
- `vessel`
- `forecast_horizon`
- `forecast_step`
- `date`
- `prediction`

De modellspesifikke filene `models/<modell>/future_predictions.csv` lagrer basisprognosen med:

- `model`
- `vessel`
- `forecast_step`
- `date`
- `prediction`
