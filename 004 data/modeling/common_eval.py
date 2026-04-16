from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.abs(y_true) + np.abs(y_pred)
    safe_ratio = np.where(
        denominator == 0.0,
        0.0,
        (200.0 * np.abs(y_pred - y_true)) / denominator,
    )
    return float(np.mean(safe_ratio))


def select_representative_vessel(panel_df: pd.DataFrame) -> str | None:
    if panel_df.empty:
        return None
    vessel_means = (
        panel_df.groupby("vessel", as_index=False)["offhire_days"]
        .mean()
        .sort_values(["offhire_days", "vessel"], ascending=[False, True])
    )
    if vessel_means.empty:
        return None
    return str(vessel_means.iloc[0]["vessel"])


def add_error_columns(pred_df: pd.DataFrame) -> pd.DataFrame:
    if pred_df.empty:
        return pred_df.copy()

    df = pred_df.copy()
    df["error"] = df["prediction"] - df["actual"]
    df["abs_error"] = df["error"].abs()
    df["squared_error"] = df["error"] ** 2
    denominator = df["actual"].abs() + df["prediction"].abs()
    df["smape_component"] = np.where(
        denominator == 0.0,
        0.0,
        (200.0 * df["abs_error"]) / denominator,
    )
    return df


def summarize_prediction_frame(pred_df: pd.DataFrame) -> tuple[float, float, float]:
    enriched = add_error_columns(pred_df)
    return (
        float(enriched["abs_error"].mean()),
        float(np.sqrt(enriched["squared_error"].mean())),
        float(enriched["smape_component"].mean()),
    )


def build_future_prediction_row(
    model: str,
    vessel: str,
    date_value: pd.Timestamp,
    prediction: float,
    forecast_step: int,
) -> dict[str, Any]:
    return {
        "model": model,
        "vessel": vessel,
        "forecast_step": forecast_step,
        "date": date_value.strftime("%Y-%m-%d"),
        "prediction": float(prediction),
    }


def build_metrics_table(
    pred_df: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    enriched = add_error_columns(pred_df)
    if enriched.empty:
        return pd.DataFrame()

    grouped = (
        enriched.groupby(group_columns, as_index=False)
        .agg(
            n_predictions=("actual", "size"),
            mae=("abs_error", "mean"),
            rmse=("squared_error", lambda values: float(np.sqrt(np.mean(values)))),
            smape=("smape_component", "mean"),
        )
        .sort_values(group_columns)
        .reset_index(drop=True)
    )
    return grouped
