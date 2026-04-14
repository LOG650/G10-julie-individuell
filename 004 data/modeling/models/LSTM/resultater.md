# LSTM resultater

Dette dokumentet genereres automatisk ved hver kjøring av `004 data/modeling/run_models.py`.

- Sist generert: `2026-04-14T18:53:23`
- Status: `ok`
- MAE: `7.7696`
- RMSE: `17.1056`
- Evaluering train: `2021-04 til 2024-12`
- Evaluering test: `2025-01 til 2026-03`
- Fremtidsprognoser trener på full historikk: `2021-04 til 2026-03`

## Detaljer

```json
{
  "evaluation_train_period": "2021-04 til 2024-12",
  "evaluation_test_period": "2025-01 til 2026-03",
  "forecast_training_period": "2021-04 til 2026-03",
  "train_observations": 675,
  "test_observations": 227,
  "forecast_observations": 902,
  "train_sequences": 630,
  "test_sequences": 225,
  "evaluation_method": "fast holdout med sekvenser og testperiode 2025-01 til 2026-03",
  "evaluation_level": "fartøynivå",
  "sequence_length": 3,
  "input_features": [
    "offhire_days",
    "month_sin",
    "month_cos",
    "special_flag"
  ],
  "architecture": {
    "lstm_units": 32,
    "dense_units": 16,
    "batch_size": 4,
    "max_epochs": 200
  }
}
```

## Prediksjoner

| model | vessel | date | actual | prediction |
| --- | --- | --- | --- | --- |
| lstm | Fartøy 1 | 2025-01-01 | 0.0000 | 0.6762 |
| lstm | Fartøy 1 | 2025-02-01 | 0.0000 | 0.0310 |
| lstm | Fartøy 1 | 2025-03-01 | 0.0000 | 0.0000 |
| lstm | Fartøy 1 | 2025-04-01 | 0.0000 | 0.0000 |
| lstm | Fartøy 1 | 2025-05-01 | 0.0000 | 0.0000 |
| lstm | Fartøy 1 | 2025-06-01 | 0.0000 | 0.0000 |
| lstm | Fartøy 1 | 2025-07-01 | 0.0000 | 0.0000 |
| lstm | Fartøy 1 | 2025-08-01 | 0.0000 | 0.0000 |
| lstm | Fartøy 1 | 2025-09-01 | 0.0000 | 0.1416 |
| lstm | Fartøy 1 | 2025-10-01 | 0.0000 | 0.1329 |

## Fremtidsprognoser

| model | vessel | date | prediction |
| --- | --- | --- | --- |
| lstm | Fartøy 1 | 2026-04-01 | 1.9751 |
| lstm | Fartøy 1 | 2026-05-01 | 1.5744 |
| lstm | Fartøy 10 | 2026-04-01 | 5.7573 |
| lstm | Fartøy 10 | 2026-05-01 | 3.6330 |
| lstm | Fartøy 11 | 2026-04-01 | 14.2982 |
| lstm | Fartøy 11 | 2026-05-01 | 9.9353 |
| lstm | Fartøy 12 | 2026-04-01 | 1.5496 |
| lstm | Fartøy 12 | 2026-05-01 | 2.1884 |
| lstm | Fartøy 13 | 2026-04-01 | 1.5496 |
| lstm | Fartøy 13 | 2026-05-01 | 2.1884 |

Viser de første 10 av totalt 225 prediksjonsrader.

Viser de første 10 av totalt 30 fremtidsprognoser.
