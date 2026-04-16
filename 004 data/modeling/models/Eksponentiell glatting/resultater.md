# Eksponentiell glatting resultater

Dette dokumentet genereres automatisk ved hver kjøring av `004 data/modeling/run_models.py`.

- Sist generert: `2026-04-16T12:31:13`
- Status: `ok`
- MAE: `8.3676`
- RMSE: `17.9544`
- sMAPE: `168.6203`
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
  "vessels_used": 15,
  "test_rows": 225,
  "evaluation_method": "ekspanderende 1-stegs prognose gjennom testperioden",
  "evaluation_level": "fartøynivå",
  "representative_vessel": "Fartøy 2",
  "selection_table": "modellvalg_per_fartoy.md",
  "residual_table": "residualdiagnostikk.md"
}
```

## Testprediksjoner

| model | vessel | date | actual | prediction |
| --- | --- | --- | --- | --- |
| exponential_smoothing | Fartøy 1 | 2025-01-01 | 0.0000 | 1.1075 |
| exponential_smoothing | Fartøy 1 | 2025-02-01 | 0.0000 | 0.9015 |
| exponential_smoothing | Fartøy 1 | 2025-03-01 | 0.0000 | 0.7382 |
| exponential_smoothing | Fartøy 1 | 2025-04-01 | 0.0000 | 0.6071 |
| exponential_smoothing | Fartøy 1 | 2025-05-01 | 0.0000 | 0.5009 |
| exponential_smoothing | Fartøy 1 | 2025-06-01 | 0.0000 | 0.4141 |
| exponential_smoothing | Fartøy 1 | 2025-07-01 | 0.0000 | 0.3430 |
| exponential_smoothing | Fartøy 1 | 2025-08-01 | 0.0000 | 0.2845 |
| exponential_smoothing | Fartøy 1 | 2025-09-01 | 0.0000 | 0.2362 |
| exponential_smoothing | Fartøy 1 | 2025-10-01 | 0.0000 | 0.1962 |

Viser de første 10 av totalt 225 testprediksjoner.

## Fremtidsprognoser

| model | vessel | forecast_step | date | prediction |
| --- | --- | --- | --- | --- |
| exponential_smoothing | Fartøy 1 | 1 | 2026-04-01 | 0.0651 |
| exponential_smoothing | Fartøy 1 | 2 | 2026-05-01 | 0.0651 |
| exponential_smoothing | Fartøy 1 | 3 | 2026-06-01 | 0.0651 |
| exponential_smoothing | Fartøy 1 | 4 | 2026-07-01 | 0.0651 |
| exponential_smoothing | Fartøy 1 | 5 | 2026-08-01 | 0.0651 |
| exponential_smoothing | Fartøy 1 | 6 | 2026-09-01 | 0.0651 |
| exponential_smoothing | Fartøy 1 | 7 | 2026-10-01 | 0.0651 |
| exponential_smoothing | Fartøy 1 | 8 | 2026-11-01 | 0.0651 |
| exponential_smoothing | Fartøy 1 | 9 | 2026-12-01 | 0.0651 |
| exponential_smoothing | Fartøy 1 | 10 | 2027-01-01 | 0.0651 |

Viser de første 10 av totalt 180 fremtidsprognoser.
