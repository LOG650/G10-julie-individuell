# XGBoost resultater

Dette dokumentet genereres automatisk ved hver kjøring av `004 data/modeling/run_models.py`.

- Sist generert: `2026-04-14T18:53:23`
- Status: `ok`
- MAE: `8.5433`
- RMSE: `18.3682`
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
  "train_rows": 630,
  "test_rows": 225,
  "evaluation_method": "fast holdout med historiske lags og testperiode 2025-01 til 2026-03",
  "evaluation_level": "fartøynivå",
  "feature_columns": [
    "month_num",
    "lag_1",
    "lag_2",
    "lag_3",
    "rolling_mean_3",
    "rolling_std_3",
    "month_sin",
    "month_cos",
    "vessel",
    "special_flag"
  ],
  "model_hyperparameters": {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9
  }
}
```

## Prediksjoner

| vessel | date | model | actual | prediction |
| --- | --- | --- | --- | --- |
| Fartøy 1 | 2025-01-01 | xgboost | 0.0000 | 1.9192 |
| Fartøy 1 | 2025-02-01 | xgboost | 0.0000 | 2.8913 |
| Fartøy 1 | 2025-03-01 | xgboost | 0.0000 | 0.3129 |
| Fartøy 1 | 2025-04-01 | xgboost | 0.0000 | 1.2894 |
| Fartøy 1 | 2025-05-01 | xgboost | 0.0000 | 1.3276 |
| Fartøy 1 | 2025-06-01 | xgboost | 0.0000 | 1.6664 |
| Fartøy 1 | 2025-07-01 | xgboost | 0.0000 | 1.1279 |
| Fartøy 1 | 2025-08-01 | xgboost | 0.0000 | 1.3505 |
| Fartøy 1 | 2025-09-01 | xgboost | 0.0000 | 0.7133 |
| Fartøy 1 | 2025-10-01 | xgboost | 0.0000 | 2.2131 |

## Fremtidsprognoser

| model | vessel | date | prediction |
| --- | --- | --- | --- |
| xgboost | Fartøy 1 | 2026-04-01 | 1.1217 |
| xgboost | Fartøy 1 | 2026-05-01 | 43.0523 |
| xgboost | Fartøy 10 | 2026-04-01 | 17.7994 |
| xgboost | Fartøy 10 | 2026-05-01 | 19.3069 |
| xgboost | Fartøy 11 | 2026-04-01 | 13.5653 |
| xgboost | Fartøy 11 | 2026-05-01 | 20.1427 |
| xgboost | Fartøy 12 | 2026-04-01 | 0.9918 |
| xgboost | Fartøy 12 | 2026-05-01 | 0.5843 |
| xgboost | Fartøy 13 | 2026-04-01 | 0.7515 |
| xgboost | Fartøy 13 | 2026-05-01 | 0.3439 |

Viser de første 10 av totalt 225 prediksjonsrader.

Viser de første 10 av totalt 30 fremtidsprognoser.
