# LSTM kode

Dette dokumentet genereres automatisk fra den aktive implementasjonen i `004 data/modeling/run_models.py`.

- Modell: `lstm`
- Funksjon: `run_lstm`

```python
def run_lstm(
    panel_df: pd.DataFrame,
    split_metadata: dict[str, Any],
) -> tuple[ModelResult, pd.DataFrame]:
    X, y, metadata = build_lstm_sequences(panel_df, window_size=3)
    dates = pd.to_datetime([item["date"] for item in metadata])
    train_mask = np.array([date <= TRAIN_END_DATE for date in dates], dtype=bool)
    test_mask = np.array([date >= TEST_START_DATE for date in dates], dtype=bool)

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise DataTooShortError("Train/test-splitt for LSTM ble tom.")

    X_train, X_test = X[train_mask], X[test_mask]
    y_test = y[test_mask]

    model, x_scaler, y_scaler = train_lstm_regressor(X_train, y[train_mask])
    X_test_scaled = x_scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    predictions_scaled = model.predict(X_test_scaled, verbose=0).reshape(-1, 1)
    predictions = y_scaler.inverse_transform(predictions_scaled).reshape(-1)
    predictions = np.clip(predictions, 0.0, None)
    if not np.isfinite(predictions).all():
        raise RuntimeError("LSTM produserte ikke-finite prediksjoner.")

    pred_records: list[dict[str, Any]] = []
    for idx, pred in zip(np.where(test_mask)[0], predictions):
        pred_records.append(
            {
                "model": "lstm",
                "vessel": metadata[idx]["vessel"],
                "date": pd.Timestamp(metadata[idx]["date"]).strftime("%Y-%m-%d"),
                "actual": float(y[idx]),
                "prediction": float(pred),
            }
        )

    pred_df = pd.DataFrame(pred_records)
    metrics = ModelResult(
        model="lstm",
        status="ok",
        mae=float(mean_absolute_error(pred_df["actual"], pred_df["prediction"])),
        rmse=rmse(pred_df["actual"].to_numpy(), pred_df["prediction"].to_numpy()),
        details={
            **split_metadata,
            "train_sequences": int(train_mask.sum()),
            "test_sequences": int(test_mask.sum()),
            "evaluation_method": "fast holdout med sekvenser og testperiode 2025-01 til 2026-03",
            "evaluation_level": "fartøynivå",
            "sequence_length": 3,
            "input_features": ["offhire_days", "month_sin", "month_cos", "special_flag"],
            "architecture": {
                "lstm_units": 32,
                "dense_units": 16,
                "batch_size": 4,
                "max_epochs": 200,
            },
        },
    )
    return metrics, pred_df
```
