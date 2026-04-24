# LSTM resultater

Dette dokumentet genereres automatisk ved hver kjøring av `004 data/modeling/run_models.py`.

- Sist generert: `2026-04-24T11:42:50`
- Status: `ok`
- MAE: `7.5722`
- RMSE: `17.4062`
- sMAPE: `178.7664`
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
  "train_sequences": 495,
  "test_sequences": 225,
  "evaluation_method": "ekspanderende 1-stegs prognose med månedlig re-trening",
  "evaluation_level": "fartøynivå",
  "walk_forward_steps": 15,
  "sequence_length": 12,
  "input_features": [
    "offhire_days",
    "month_sin",
    "month_cos",
    "special_flag"
  ],
  "representative_vessel": "Fartøy 2",
  "architecture": {
    "lstm_units": 32,
    "dense_units": 16,
    "batch_size": 8,
    "max_epochs": 100
  }
}
```

## Testprediksjoner

| model | vessel | date | actual | prediction |
| --- | --- | --- | --- | --- |
| lstm | Fartøy 1 | 2025-01-01 | 0.0000 | 0.0000 |
| lstm | Fartøy 10 | 2025-01-01 | 0.0000 | 6.4530 |
| lstm | Fartøy 11 | 2025-01-01 | 0.0000 | 9.6724 |
| lstm | Fartøy 12 | 2025-01-01 | 0.0000 | 3.4918 |
| lstm | Fartøy 13 | 2025-01-01 | 0.0000 | 3.3053 |
| lstm | Fartøy 14 | 2025-01-01 | 0.0000 | 3.3053 |
| lstm | Fartøy 15 | 2025-01-01 | 19.8900 | 3.3053 |
| lstm | Fartøy 2 | 2025-01-01 | 0.0000 | 5.6822 |
| lstm | Fartøy 3 | 2025-01-01 | 0.0000 | 6.6883 |
| lstm | Fartøy 4 | 2025-01-01 | 0.0000 | 4.5207 |

Viser de første 10 av totalt 225 testprediksjoner.

## Fremtidsprognoser

| model | vessel | forecast_step | date | prediction |
| --- | --- | --- | --- | --- |
| lstm | Fartøy 1 | 1 | 2026-04-01 | 0.0000 |
| lstm | Fartøy 1 | 2 | 2026-05-01 | 0.0000 |
| lstm | Fartøy 1 | 3 | 2026-06-01 | 0.0000 |
| lstm | Fartøy 1 | 4 | 2026-07-01 | 0.0000 |
| lstm | Fartøy 1 | 5 | 2026-08-01 | 0.0000 |
| lstm | Fartøy 1 | 6 | 2026-09-01 | 0.0000 |
| lstm | Fartøy 1 | 7 | 2026-10-01 | 0.0000 |
| lstm | Fartøy 1 | 8 | 2026-11-01 | 0.0000 |
| lstm | Fartøy 1 | 9 | 2026-12-01 | 0.0000 |
| lstm | Fartøy 1 | 10 | 2027-01-01 | 0.0000 |

Viser de første 10 av totalt 180 fremtidsprognoser.
