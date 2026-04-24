# XGBoost resultater

Dette dokumentet genereres automatisk ved hver kjøring av `004 data/modeling/run_models.py`.

- Sist generert: `2026-04-24T11:42:50`
- Status: `ok`
- MAE: `7.3487`
- RMSE: `17.4010`
- sMAPE: `182.9827`
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
  "train_rows": 495,
  "test_rows": 225,
  "walk_forward_steps": 15,
  "evaluation_method": "ekspanderende 1-stegs prognose med månedlig re-trening",
  "evaluation_level": "fartøynivå",
  "feature_columns": [
    "month_num",
    "quarter_num",
    "year_num",
    "time_idx",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_6",
    "lag_12",
    "rolling_mean_3",
    "rolling_mean_6",
    "rolling_mean_12",
    "rolling_std_3",
    "rolling_std_6",
    "rolling_std_12",
    "month_sin",
    "month_cos",
    "vessel",
    "special_flag"
  ],
  "representative_vessel": "Fartøy 2",
  "model_hyperparameters": {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9
  }
}
```

## Testprediksjoner

| model | vessel | date | actual | prediction |
| --- | --- | --- | --- | --- |
| xgboost | Fartøy 1 | 2025-01-01 | 0.0000 | 1.0546 |
| xgboost | Fartøy 10 | 2025-01-01 | 0.0000 | 0.0000 |
| xgboost | Fartøy 11 | 2025-01-01 | 0.0000 | 6.4019 |
| xgboost | Fartøy 12 | 2025-01-01 | 0.0000 | 1.3956 |
| xgboost | Fartøy 13 | 2025-01-01 | 0.0000 | 0.6748 |
| xgboost | Fartøy 14 | 2025-01-01 | 0.0000 | 1.1317 |
| xgboost | Fartøy 15 | 2025-01-01 | 19.8900 | 1.2228 |
| xgboost | Fartøy 2 | 2025-01-01 | 0.0000 | 1.7476 |
| xgboost | Fartøy 3 | 2025-01-01 | 0.0000 | 0.4061 |
| xgboost | Fartøy 4 | 2025-01-01 | 0.0000 | 11.3168 |

Viser de første 10 av totalt 225 testprediksjoner.

## Fremtidsprognoser

| model | vessel | forecast_step | date | prediction |
| --- | --- | --- | --- | --- |
| xgboost | Fartøy 1 | 1 | 2026-04-01 | 0.7973 |
| xgboost | Fartøy 1 | 2 | 2026-05-01 | 0.0000 |
| xgboost | Fartøy 1 | 3 | 2026-06-01 | 5.8297 |
| xgboost | Fartøy 1 | 4 | 2026-07-01 | 4.9000 |
| xgboost | Fartøy 1 | 5 | 2026-08-01 | 2.8385 |
| xgboost | Fartøy 1 | 6 | 2026-09-01 | 3.8629 |
| xgboost | Fartøy 1 | 7 | 2026-10-01 | 5.3420 |
| xgboost | Fartøy 1 | 8 | 2026-11-01 | 7.4683 |
| xgboost | Fartøy 1 | 9 | 2026-12-01 | 17.7179 |
| xgboost | Fartøy 1 | 10 | 2027-01-01 | 28.8079 |

Viser de første 10 av totalt 180 fremtidsprognoser.
