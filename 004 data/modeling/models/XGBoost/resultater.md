# XGBoost resultater

Dette dokumentet genereres automatisk ved hver kjøring av `004 data/modeling/run_models.py`.

- Sist generert: `2026-04-15T15:14:12`
- Status: `ok`
- MAE: `7.3487`
- RMSE: `17.4010`
- sMAPE: `182.9827`
- Evaluering train: `2021-04 til 2024-12`
- Evaluering test: `2025-01 til 2026-03`

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
