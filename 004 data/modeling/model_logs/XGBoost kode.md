# XGBoost kode

Dette dokumentet genereres automatisk fra den aktive implementasjonen i `004 data/modeling/run_models.py`.

- Modell: `xgboost`
- Funksjon: `run_xgboost`

```python
def run_xgboost(features_df: pd.DataFrame) -> tuple[ModelResult, pd.DataFrame]:
    from xgboost import XGBRegressor

    train_df, test_df = train_test_split_panel(features_df)
    numeric_features = [
        "month_num",
        "lag_1",
        "lag_2",
        "lag_3",
        "rolling_mean_3",
        "rolling_std_3",
        "month_sin",
        "month_cos",
    ]
    categorical_features = ["vessel", "special_flag"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ]
    )

    model = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    feature_columns = numeric_features + categorical_features
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
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
        },
    )
    return metrics, pred_df
```
