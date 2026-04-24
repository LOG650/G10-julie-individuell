# Modellering

Denne mappen inneholder modelleringskoden for rapporten og en tydelig skilt output-struktur.

Målet er å sammenligne fire modeller mot samme problemstilling:

- `SARIMA`
- `Eksponentiell glatting`
- `XGBoost`
- `LSTM`

## Struktur

- `../raw/`: anonymisert masterfil
- `../processed/`: eksplisitt train/test-splitt brukt av pipeline
- `./`: modelleringskode og avhengigheter
- `outputs/shared/`: samlede testresultater, fremtidsprognoser og figurer
- `outputs/models/`: modellspesifikke artefakter
- `outputs/logs/`: samlet modellsammenligning

## Datagrunnlag

Pipelinen bruker `004 data/raw/Data som skal brukes Anonymisert.csv` som masterfil.
Fra denne genereres to eksplisitte datasett i `004 data/processed/`:

- `train.csv`
- `test.csv`

Disse brukes til historisk evaluering med en fast tidsbasert splitt:

- `train`: april 2021 til desember 2024
- `test`: januar 2025 til mars 2026

Standardkjøringen dekker to faser i samme pipeline:

- historisk modellbygging, testing og verifikasjon
- fremtidsprognoser på full historikk

## Oppsett

Det virtuelle miljøet skal ligge utenfor repoet. På denne maskinen brukes:
`/Users/juliekarlsen/.venvs/g10-julie-individuell`

Aktiver miljøet:

```bash
source /Users/juliekarlsen/.venvs/g10-julie-individuell/bin/activate
```

Hvis miljøet må opprettes på nytt:

```bash
python3 -m venv /Users/juliekarlsen/.venvs/g10-julie-individuell
source /Users/juliekarlsen/.venvs/g10-julie-individuell/bin/activate
pip install -r '004 data/modeling/requirements.txt'
```

## Kjøring

Generer først train/test-filene:

```bash
python '004 data/modeling/generate_train_test_split.py'
```

Kjør deretter hele modellpakken:

```bash
python '004 data/modeling/run_models.py'
```

Skriptet oppdaterer `004 data/processed/train.csv` og `004 data/processed/test.csv` fra masterfilen ved hver kjøring.

## Output

Samlede artefakter skrives til `004 data/modeling/outputs/shared/`, blant annet:

- `metrics.json`
- `predictions.csv`
- `model_comparison_summary.csv`
- `metrics_by_vessel.csv`
- `metrics_by_month.csv`
- `future_predictions.csv`
- `future_totals_by_model_and_date.csv`
- `future_predictions_1m.csv`, `3m.csv`, `6m.csv`, `12m.csv`
- `future_predictions_1m_pivot.csv`, `3m_pivot.csv`, `6m_pivot.csv`, `12m_pivot.csv`
- `figures/mae_per_model.png`
- `figures/mae_by_month.png`
- `figures/mae_heatmap_by_vessel.png`
- `figures/future_total_offhire_1m.png`
- `figures/future_total_offhire_3m.png`
- `figures/future_total_offhire_6m.png`
- `figures/future_total_offhire_12m.png`

Modellspesifikke artefakter skrives til `004 data/modeling/outputs/models/<modell>/`, inkludert:

- `metrics.json`
- `predictions.csv`
- `future_predictions.csv`
- `metode.md`
- `kode.md`
- `resultater.md`

I tillegg genereres en samlet sammenligningslogg i:

- `004 data/modeling/outputs/logs/Modellsammenligning.md`

## Evalueringslogikk

Alle fire modellene evalueres under samme hovedoppsett:

- målnivå: månedlig `offhire_days` per fartøy
- train: `2021-04` til `2024-12`
- test: `2025-01` til `2026-03`
- metode: ekspanderende `1-stegs` prognose gjennom testperioden
- hovedmål: `MAE`
- støttemål: `RMSE` og `sMAPE`

## Fremtidsprognoser

Når historisk evaluering er gjennomført, brukes hele datasettet til og med `2026-03` til å bygge fremtidsprognoser.
Disse lagres for fire horisonter:

- `1` måned fram
- `3` måneder fram
- `6` måneder fram
- `12` måneder fram

Felles kolonner i de samlede forecast-filene under `outputs/shared/` er:

- `model`
- `vessel`
- `forecast_horizon`
- `forecast_step`
- `date`
- `prediction`

De modellspesifikke basisprognosene i `outputs/models/<modell>/future_predictions.csv` lagrer:

- `model`
- `vessel`
- `forecast_step`
- `date`
- `prediction`
