# Eksponentiell glatting resultater

Dette dokumentet genereres automatisk ved hver kjøring av `004 data/modeling/run_models.py`.

- Sist generert: `2026-04-14T18:53:23`
- Status: `ok`
- MAE: `6.7212`
- RMSE: `17.5192`
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
  "vessels_used": 15,
  "test_rows": 225,
  "evaluation_method": "ekspanderende 1-stegs prognose gjennom testperioden",
  "evaluation_level": "fartøynivå",
  "step_2_summary": "Seriene er korte, nulltunge og ujevne, så sesongkomponent ble ikke brukt.",
  "step_3_summary": "Ikke-sesongbasert eksponentiell glatting per fartøy; additiv trend brukes bare når train-serien er lang nok og ikke konstant."
}
```

## Prediksjoner

| model | vessel | date | actual | prediction |
| --- | --- | --- | --- | --- |
| exponential_smoothing | Fartøy 1 | 2025-01-01 | 0.0000 | 2.8545 |
| exponential_smoothing | Fartøy 1 | 2025-02-01 | 0.0000 | 2.3942 |
| exponential_smoothing | Fartøy 1 | 2025-03-01 | 0.0000 | 1.9704 |
| exponential_smoothing | Fartøy 1 | 2025-04-01 | 0.0000 | 1.5798 |
| exponential_smoothing | Fartøy 1 | 2025-05-01 | 0.0000 | 1.2194 |
| exponential_smoothing | Fartøy 1 | 2025-06-01 | 0.0000 | 0.8865 |
| exponential_smoothing | Fartøy 1 | 2025-07-01 | 0.0000 | 0.5788 |
| exponential_smoothing | Fartøy 1 | 2025-08-01 | 0.0000 | 0.2941 |
| exponential_smoothing | Fartøy 1 | 2025-09-01 | 0.0000 | 0.0305 |
| exponential_smoothing | Fartøy 1 | 2025-10-01 | 0.0000 | 0.0000 |

## Fremtidsprognoser

| model | vessel | date | prediction |
| --- | --- | --- | --- |
| exponential_smoothing | Fartøy 1 | 2026-04-01 | 0.0000 |
| exponential_smoothing | Fartøy 1 | 2026-05-01 | 0.0000 |
| exponential_smoothing | Fartøy 10 | 2026-04-01 | 25.7160 |
| exponential_smoothing | Fartøy 10 | 2026-05-01 | 26.0062 |
| exponential_smoothing | Fartøy 11 | 2026-04-01 | 16.0662 |
| exponential_smoothing | Fartøy 11 | 2026-05-01 | 16.3809 |
| exponential_smoothing | Fartøy 12 | 2026-04-01 | 0.3423 |
| exponential_smoothing | Fartøy 12 | 2026-05-01 | 0.3237 |
| exponential_smoothing | Fartøy 13 | 2026-04-01 | 0.0000 |
| exponential_smoothing | Fartøy 13 | 2026-05-01 | 0.0000 |

Viser de første 10 av totalt 225 prediksjonsrader.

Viser de første 10 av totalt 30 fremtidsprognoser.
