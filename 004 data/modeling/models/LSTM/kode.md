# LSTM kode

Dette dokumentet genereres automatisk fra den aktive implementasjonen i `004 data/modeling/run_models.py`.

- Modell: `lstm`
- Funksjon: `run_lstm`

```python
def run_lstm(
    panel_df: pd.DataFrame,
    split_metadata: dict[str, Any],
) -> tuple[ModelResult, pd.DataFrame]:
    history_panel = panel_df[panel_df["date"] <= TRAIN_END_DATE].copy()
    test_panel = panel_df[panel_df["date"] >= TEST_START_DATE].copy()
    representative_vessel = select_representative_vessel(history_panel)
    window_size = 12

    X_reference, y_reference, _ = build_lstm_sequences(history_panel, window_size=window_size)
    _, _, _, history_df = train_lstm_regressor(X_reference, y_reference)
    save_lstm_training_history(history_df)

    pred_records: list[dict[str, Any]] = []
    test_dates = sorted(test_panel["date"].unique())

    for date_value in test_dates:
        X_train, y_train, _ = build_lstm_sequences(history_panel, window_size=window_size)
        if len(X_train) == 0:
            continue

        model, x_scaler, y_scaler, _ = train_lstm_regressor(X_train, y_train)
        month_df = test_panel[test_panel["date"] == date_value].copy()

        for _, row in month_df.iterrows():
            vessel_history = (
                history_panel[history_panel["vessel"] == row["vessel"]]
                .sort_values("date")
                .reset_index(drop=True)
            )
            if len(vessel_history) < window_size:
                continue

            history_values = vessel_history["offhire_days"].to_numpy(dtype=np.float32)[-window_size:]
            history_months = vessel_history["month_num"].to_numpy(dtype=np.float32)[-window_size:]
            history_special = (
                vessel_history["Spesielle behov/krav"]
                .fillna("")
                .str.strip()
                .ne("")
                .astype(np.float32)
                .to_numpy()[-window_size:]
            )
            sequence = np.stack(
                [
                    history_values,
                    np.sin(2 * np.pi * history_months / 12.0),
                    np.cos(2 * np.pi * history_months / 12.0),
                    history_special,
                ],
                axis=1,
            )
            sequence_scaled = x_scaler.transform(sequence).reshape(1, sequence.shape[0], sequence.shape[1])
            prediction_scaled = model.predict(sequence_scaled, verbose=0).reshape(-1, 1)
            prediction = float(
                np.clip(
                    y_scaler.inverse_transform(prediction_scaled).reshape(-1)[0],
                    0.0,
                    MAX_OFFHIRE_VALUE,
                )
            )
            pred_records.append(
                {
                    "model": "lstm",
                    "vessel": row["vessel"],
                    "date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
                    "actual": float(row["offhire_days"]),
                    "prediction": prediction,
                }
            )

        history_panel = pd.concat([history_panel, month_df], ignore_index=True)
        history_panel = history_panel.sort_values(["vessel", "date"]).reset_index(drop=True)

    pred_df = pd.DataFrame(pred_records)
    save_representative_prediction_plot(
        "lstm",
        representative_vessel,
        panel_df[panel_df["date"] <= TRAIN_END_DATE],
        test_panel,
        pred_df,
        "LSTM: representativt testforløp",
    )
    mae_value, rmse_value, smape_value = summarize_prediction_frame(pred_df)
    metrics = ModelResult(
        model="lstm",
        status="ok",
        mae=mae_value,
        rmse=rmse_value,
        smape=smape_value,
        details={
            **split_metadata,
            "train_sequences": int(len(X_reference)),
            "test_sequences": int(len(pred_df)),
            "evaluation_method": "ekspanderende 1-stegs prognose med månedlig re-trening",
            "evaluation_level": "fartøynivå",
            "walk_forward_steps": int(len(test_dates)),
            "sequence_length": window_size,
            "input_features": ["offhire_days", "month_sin", "month_cos", "special_flag"],
            "representative_vessel": representative_vessel,
            "architecture": {
                "lstm_units": 32,
                "dense_units": 16,
                "batch_size": 8,
                "max_epochs": 100,
            },
        },
    )
    return metrics, pred_df
```
