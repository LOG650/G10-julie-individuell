# XGBoost kode

Dette dokumentet genereres automatisk fra den aktive implementasjonen i `004 data/modeling/xgboost_model.py`.

- Modell: `xgboost`
- Funksjon: `run_xgboost`

```python
def run_xgboost(
    panel_df: pd.DataFrame,
    split_metadata: dict[str, Any],
) -> tuple[ModelResult, pd.DataFrame]:
    history_panel = panel_df[panel_df["date"] <= TRAIN_END_DATE].copy()
    test_panel = panel_df[panel_df["date"] >= TEST_START_DATE].copy()
    reference_train_df = build_panel_features(history_panel)
    if reference_train_df.empty:
        raise DataTooShortError("For få historiske observasjoner til å bygge XGBoost-features.")

    reference_pipeline, feature_columns = build_xgboost_pipeline()
    reference_pipeline.fit(reference_train_df[feature_columns], reference_train_df["offhire_days"])
    save_feature_importance_artifacts(reference_pipeline)

    representative_vessel = select_representative_vessel(history_panel)
    predictions: list[dict[str, Any]] = []
    test_dates = sorted(test_panel["date"].unique())

    for date_value in test_dates:
        train_df = build_panel_features(history_panel)
        if train_df.empty:
            continue
        target_month_df = test_panel[test_panel["date"] == date_value].copy()
        feature_rows = build_xgboost_feature_rows(history_panel, target_month_df)
        if feature_rows.empty:
            history_panel = pd.concat([history_panel, target_month_df], ignore_index=True)
            history_panel = history_panel.sort_values(["vessel", "date"]).reset_index(drop=True)
            continue

        pipeline, feature_columns = build_xgboost_pipeline()
        pipeline.fit(train_df[feature_columns], train_df["offhire_days"])
        step_predictions = np.clip(
            pipeline.predict(feature_rows[feature_columns]),
            0.0,
            MAX_OFFHIRE_VALUE,
        )
        for _, row, prediction in zip(feature_rows.index, feature_rows.itertuples(index=False), step_predictions):
            predictions.append(
                {
                    "model": "xgboost",
                    "vessel": row.vessel,
                    "date": pd.Timestamp(row.date).strftime("%Y-%m-%d"),
                    "actual": float(row.actual),
                    "prediction": float(prediction),
                }
            )

        history_panel = pd.concat([history_panel, target_month_df], ignore_index=True)
        history_panel = history_panel.sort_values(["vessel", "date"]).reset_index(drop=True)

    pred_df = pd.DataFrame(predictions)
    save_representative_prediction_plot(
        "xgboost",
        representative_vessel,
        panel_df[panel_df["date"] <= TRAIN_END_DATE],
        test_panel,
        pred_df,
        "XGBoost: representativt testforløp",
    )
    mae_value, rmse_value, smape_value = summarize_prediction_frame(pred_df)
    metrics = ModelResult(
        model="xgboost",
        status="ok",
        mae=mae_value,
        rmse=rmse_value,
        smape=smape_value,
        details={
            **split_metadata,
            "train_rows": int(len(reference_train_df)),
            "test_rows": int(len(pred_df)),
            "walk_forward_steps": int(len(test_dates)),
            "evaluation_method": "ekspanderende 1-stegs prognose med månedlig re-trening",
            "evaluation_level": "fartøynivå",
            "feature_columns": feature_columns,
            "representative_vessel": representative_vessel,
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
