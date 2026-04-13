# LSTM kode

Dette dokumentet genereres automatisk fra den aktive implementasjonen i `004 data/modeling/run_models.py`.

- Modell: `lstm`
- Funksjon: `run_lstm`

```python
def run_lstm(panel_df: pd.DataFrame) -> tuple[ModelResult, pd.DataFrame]:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    from tensorflow import keras

    X, y, metadata = build_lstm_sequences(panel_df, window_size=3)
    dates = pd.to_datetime([item["date"] for item in metadata])
    unique_dates = sorted(dates.unique())
    if len(unique_dates) < 4:
        raise DataTooShortError("For få datoer til å evaluere LSTM-modellen.")

    test_dates = set(unique_dates[-2:])
    test_mask = np.array([date in test_dates for date in dates], dtype=bool)
    train_mask = ~test_mask

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise DataTooShortError("Train/test-splitt for LSTM ble tom.")

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    x_scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    X_test_flat = X_test.reshape(-1, X_test.shape[-1])
    x_scaler.fit(X_train_flat)
    X_train_scaled = x_scaler.transform(X_train_flat).reshape(X_train.shape)
    X_test_scaled = x_scaler.transform(X_test_flat).reshape(X_test.shape)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1)

    tf.keras.utils.set_random_seed(42)
    np.random.seed(42)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
            keras.layers.LSTM(32, activation="tanh"),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
        loss="mse",
    )
    callback = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
    )
    model.fit(
        X_train_scaled,
        y_train_scaled,
        validation_split=0.2,
        epochs=200,
        batch_size=4,
        verbose=0,
        shuffle=False,
        callbacks=[callback],
    )
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
            "train_sequences": int(train_mask.sum()),
            "test_sequences": int(test_mask.sum()),
        },
    )
    return metrics, pred_df
```
