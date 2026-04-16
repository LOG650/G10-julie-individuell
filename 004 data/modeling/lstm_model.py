from __future__ import annotations

import os
from typing import Any

from common_config import (
    MAX_FUTURE_FORECAST_HORIZON,
    MAX_OFFHIRE_VALUE,
    TEST_START_DATE,
    TRAIN_END_DATE,
)
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from common_data import future_dates_from_panel
from common_eval import (
    build_future_prediction_row,
    select_representative_vessel,
    summarize_prediction_frame,
)
from common_io import save_lstm_training_history, save_representative_prediction_plot
from common_types import DataTooShortError, ModelResult


def build_lstm_sequences(
    panel_df: pd.DataFrame,
    window_size: int = 12,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    sequences: list[np.ndarray] = []
    targets: list[float] = []
    metadata: list[dict[str, Any]] = []

    for vessel, vessel_df in panel_df.groupby("vessel"):
        vessel_df = vessel_df.sort_values("date").reset_index(drop=True)
        values = vessel_df["offhire_days"].to_numpy(dtype=np.float32)
        month_num = vessel_df["month_num"].to_numpy(dtype=np.float32)
        special_flag = vessel_df["Spesielle behov/krav"].fillna("").str.strip().ne("").astype(np.float32).to_numpy()

        if len(vessel_df) <= window_size:
            continue

        for idx in range(window_size, len(vessel_df)):
            history = values[idx - window_size : idx]
            history_months = month_num[idx - window_size : idx]
            history_special = special_flag[idx - window_size : idx]
            sequence = np.stack(
                [
                    history,
                    np.sin(2 * np.pi * history_months / 12.0),
                    np.cos(2 * np.pi * history_months / 12.0),
                    history_special,
                ],
                axis=1,
            )
            sequences.append(sequence)
            targets.append(values[idx])
            metadata.append(
                {
                    "vessel": vessel,
                    "date": vessel_df.loc[idx, "date"],
                }
            )

    if not sequences:
        raise DataTooShortError("Ingen LSTM-sekvenser kunne bygges fra datasettet.")

    return np.asarray(sequences, dtype=np.float32), np.asarray(targets, dtype=np.float32), metadata


def train_lstm_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[Any, StandardScaler, StandardScaler, pd.DataFrame]:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    from tensorflow import keras

    x_scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    x_scaler.fit(X_train_flat)
    X_train_scaled = x_scaler.transform(X_train_flat).reshape(X_train.shape)

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
    fit_kwargs = {
        "x": X_train_scaled,
        "y": y_train_scaled,
        "epochs": 100,
        "batch_size": 8,
        "verbose": 0,
        "shuffle": False,
    }
    if len(X_train_scaled) >= 10:
        fit_kwargs["validation_split"] = 0.2
        fit_kwargs["callbacks"] = [callback]

    history = model.fit(**fit_kwargs)
    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    return model, x_scaler, y_scaler, history_df


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


def forecast_lstm(
    panel_df: pd.DataFrame,
    horizon: int = MAX_FUTURE_FORECAST_HORIZON,
) -> pd.DataFrame:
    window_size = 12
    X, y, _ = build_lstm_sequences(panel_df, window_size=window_size)
    model, x_scaler, y_scaler, _ = train_lstm_regressor(X, y)
    future_dates = future_dates_from_panel(panel_df, horizon)
    predictions: list[dict[str, Any]] = []

    for vessel, vessel_df in panel_df.groupby("vessel"):
        vessel_df = vessel_df.sort_values("date").reset_index(drop=True)
        if len(vessel_df) <= window_size:
            continue

        history_values = vessel_df["offhire_days"].astype(float).tolist()
        history_months = vessel_df["month_num"].astype(float).tolist()
        history_special = (
            vessel_df["Spesielle behov/krav"]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
            .astype(float)
            .tolist()
        )
        latest_special_flag = history_special[-1] if history_special else 0.0

        for forecast_step, date_value in enumerate(future_dates, start=1):
            sequence = np.stack(
                [
                    np.asarray(history_values[-window_size:], dtype=np.float32),
                    np.sin(2 * np.pi * np.asarray(history_months[-window_size:], dtype=np.float32) / 12.0),
                    np.cos(2 * np.pi * np.asarray(history_months[-window_size:], dtype=np.float32) / 12.0),
                    np.asarray(history_special[-window_size:], dtype=np.float32),
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
            predictions.append(
                build_future_prediction_row(
                    "lstm",
                    vessel,
                    date_value,
                    prediction,
                    forecast_step,
                )
            )
            history_values.append(prediction)
            history_months.append(float(date_value.month))
            history_special.append(float(latest_special_flag))

    if not predictions:
        return pd.DataFrame()

    return pd.DataFrame(predictions).sort_values(["vessel", "date"]).reset_index(drop=True)
