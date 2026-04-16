# SARIMA resultater

Dette dokumentet genereres automatisk ved hver kjøring av `004 data/modeling/run_models.py`.

- Sist generert: `2026-04-16T12:31:13`
- Status: `ok`
- MAE: `6.1521`
- RMSE: `16.7823`
- sMAPE: `100.4363`
- Evaluering train: `2021-04 til 2024-12`
- Evaluering test: `2025-01 til 2026-03`
- Fremtidsprognoser: `12` steg fra `2026-04` til `2027-03`

## Detaljer

```json
{
  "evaluation_train_period": "2021-04 til 2024-12",
  "evaluation_test_period": "2025-01 til 2026-03",
  "train_observations": 675,
  "test_observations": 227,
  "series_type": "per_vessel",
  "test_rows": 225,
  "evaluation_method": "ekspanderende 1-stegs prognose per fartøy",
  "evaluation_level": "fartøynivå",
  "modeled_vessels": 15,
  "representative_vessel": "Fartøy 2",
  "artifact_files": {
    "stasjonaritet": "stasjonaritet.md",
    "kandidatmodeller": "kandidatmodeller.md",
    "modellvalg_per_fartoy": "modellvalg_per_fartoy.md",
    "residualdiagnostikk_tabell": "residualdiagnostikk.md",
    "acf": "acf.png",
    "pacf": "pacf.png",
    "residualdiagnostikk_figur": "residualdiagnostikk.png",
    "representativ_testplot": "representativ_testplot.png"
  }
}
```

## Testprediksjoner

| model | vessel | date | actual | prediction |
| --- | --- | --- | --- | --- |
| sarima | Fartøy 1 | 2025-01-01 | 0.0000 | 7.8268 |
| sarima | Fartøy 1 | 2025-02-01 | 0.0000 | 0.0000 |
| sarima | Fartøy 1 | 2025-03-01 | 0.0000 | 4.1726 |
| sarima | Fartøy 1 | 2025-04-01 | 0.0000 | 1.9153 |
| sarima | Fartøy 1 | 2025-05-01 | 0.0000 | 2.6932 |
| sarima | Fartøy 1 | 2025-06-01 | 0.0000 | 8.5243 |
| sarima | Fartøy 1 | 2025-07-01 | 0.0000 | 0.0000 |
| sarima | Fartøy 1 | 2025-08-01 | 0.0000 | 0.0000 |
| sarima | Fartøy 1 | 2025-09-01 | 0.0000 | 2.1138 |
| sarima | Fartøy 1 | 2025-10-01 | 0.0000 | 0.0000 |

Viser de første 10 av totalt 225 testprediksjoner.

## Fremtidsprognoser

| model | vessel | forecast_step | date | prediction |
| --- | --- | --- | --- | --- |
| sarima | Fartøy 1 | 1 | 2026-04-01 | 0.4575 |
| sarima | Fartøy 1 | 2 | 2026-05-01 | 0.9667 |
| sarima | Fartøy 1 | 3 | 2026-06-01 | 1.3506 |
| sarima | Fartøy 1 | 4 | 2026-07-01 | 0.0065 |
| sarima | Fartøy 1 | 5 | 2026-08-01 | 1.5304 |
| sarima | Fartøy 1 | 6 | 2026-09-01 | 0.0000 |
| sarima | Fartøy 1 | 7 | 2026-10-01 | 0.0000 |
| sarima | Fartøy 1 | 8 | 2026-11-01 | 0.0000 |
| sarima | Fartøy 1 | 9 | 2026-12-01 | 0.0000 |
| sarima | Fartøy 1 | 10 | 2027-01-01 | 0.2047 |

Viser de første 10 av totalt 180 fremtidsprognoser.
