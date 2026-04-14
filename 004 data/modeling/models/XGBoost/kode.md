# XGBoost kode

Dette dokumentet genereres automatisk fra den aktive implementasjonen i `004 data/modeling/run_models.py`.

- Modell: `xgboost`
- Funksjon: `run_xgboost`

```python
def run_xgboost(
    features_df: pd.DataFrame,
    split_metadata: dict[str, Any],
) -> tuple[ModelResult, pd.DataFrame]:
    train_df, test_df = split_features_for_evaluation(features_df)
    pipeline, feature_columns = build_xgboost_pipeline()
    pipeline.fit(train_df[feature_columns], train_df["offhire_days"])
    predictions = np.clip(pipeline.predict(test_df[feature_columns]), 0.0, None)

    pred_df = test_df[["vessel", "date", "offhire_days"]].copy()
    pred_df["model"] = "xgboost"
    pred_df["actual"] = pred_df["offhire_days"].astype(float)
    pred_df["prediction"] = predictions.astype(float)
    pred_df = pred_df.drop(columns=["offhire_days"])
    pred_df["date"] = pred_df["date"].dt.strftime("%Y-%m-%d")

    metrics = ModelResult(
        model="xgboost",
        status="ok",
        mae=float(mean_absolute_error(pred_df["actual"], pred_df["prediction"])),
        rmse=rmse(pred_df["actual"].to_numpy(), pred_df["prediction"].to_numpy()),
        details={
            **split_metadata,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "evaluation_method": "fast holdout med historiske lags og testperiode 2025-01 til 2026-03",
            "evaluation_level": "fartøynivå",
            "feature_columns": feature_columns,
            "model_hyperparameters": {
                "n_estimators": 200,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
            },
        },
    )
    return metrics, pred_df
```
