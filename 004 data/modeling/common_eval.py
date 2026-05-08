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


def build_mase_scale_map(train_panel: pd.DataFrame) -> dict[str, float]:
    if train_panel.empty:
        return {}

    scale_map: dict[str, float] = {}
    ordered = train_panel.sort_values(["vessel", "date"]).copy()
    for vessel, vessel_df in ordered.groupby("vessel"):
        values = vessel_df["offhire_days"].to_numpy(dtype=float)
        if len(values) < 2:
            continue
        scale = float(np.abs(np.diff(values)).mean())
        if scale > 0.0:
            scale_map[str(vessel)] = scale
    return scale_map


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


def add_error_columns(
    pred_df: pd.DataFrame,
    mase_scale_map: dict[str, float] | None = None,
) -> pd.DataFrame:
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
    if mase_scale_map is None:
        df["mase_scale"] = np.nan
        df["mase_component"] = np.nan
        return df

    df["mase_scale"] = pd.to_numeric(df["vessel"].map(mase_scale_map), errors="coerce")
    valid_scale_mask = df["mase_scale"].notna() & (df["mase_scale"] > 0.0)
    df["mase_component"] = np.where(
        valid_scale_mask,
        df["abs_error"] / df["mase_scale"],
        np.nan,
    )
    return df


def summarize_prediction_frame(pred_df: pd.DataFrame) -> tuple[float, float, float]:
    enriched = add_error_columns(pred_df)
    return (
        float(enriched["abs_error"].mean()),
        float(np.sqrt(enriched["squared_error"].mean())),
        float(enriched["smape_component"].mean()),
    )


def mase_from_prediction_frame(
    pred_df: pd.DataFrame,
    mase_scale_map: dict[str, float] | None = None,
) -> float | None:
    enriched = add_error_columns(pred_df, mase_scale_map=mase_scale_map)
    if "mase_component" not in enriched or enriched["mase_component"].dropna().empty:
        return None
    return float(enriched["mase_component"].mean())


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
    mase_scale_map: dict[str, float] | None = None,
) -> pd.DataFrame:
    enriched = add_error_columns(pred_df, mase_scale_map=mase_scale_map)
    if enriched.empty:
        return pd.DataFrame()

    grouped = (
        enriched.groupby(group_columns, as_index=False)
        .agg(
            n_predictions=("actual", "size"),
            mae=("abs_error", "mean"),
            rmse=("squared_error", lambda values: float(np.sqrt(np.mean(values)))),
            smape=("smape_component", "mean"),
            mase=("mase_component", "mean"),
        )
        .sort_values(group_columns)
        .reset_index(drop=True)
    )
    return grouped
