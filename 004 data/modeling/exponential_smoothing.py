from __future__ import annotations

import warnings
from typing import Any

from common_config import MAX_FUTURE_FORECAST_HORIZON, MAX_OFFHIRE_VALUE
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from common_data import build_vessel_series, future_dates_from_panel
from common_eval import (
    build_future_prediction_row,
    select_representative_vessel,
    summarize_prediction_frame,
)
from common_io import (
    model_artifact_path,
    normalize_optional_string,
    save_representative_prediction_plot,
    write_dataframe_artifacts,
)
from common_types import DataTooShortError, ModelResult


def fit_ets_model(
    train_series: pd.Series,
    trend: str | None,
    seasonal: str | None,
    seasonal_periods: int | None = None,
) -> Any:
    model = ExponentialSmoothing(
        train_series,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
        initialization_method="estimated",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return model.fit(optimized=True, use_brute=True)


def fit_or_fallback_exponential_forecast(
    train_series: pd.Series,
    steps: int,
    trend: str | None,
    seasonal: str | None,
    seasonal_periods: int | None = None,
) -> np.ndarray:
    try:
        fit = fit_ets_model(
            train_series,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
        )
        forecast = np.asarray(fit.forecast(steps), dtype=float)
        if not np.isfinite(forecast).all():
            raise ValueError("Forecast contains non-finite values.")
    except Exception:
        forecast = np.repeat(float(train_series.iloc[-1]), steps)

    return np.clip(forecast, 0.0, MAX_OFFHIRE_VALUE)


def fit_ets_candidates(train_series: pd.Series) -> pd.DataFrame:
    candidate_specs = [
        {
            "spec": "ANN",
            "trend": None,
            "seasonal": None,
            "seasonal_periods": None,
        },
        {
            "spec": "AAN",
            "trend": "add",
            "seasonal": None,
            "seasonal_periods": None,
        },
    ]
    if len(train_series) >= 24 and train_series.nunique() > 1:
        candidate_specs.append(
            {
                "spec": "AAA",
                "trend": "add",
                "seasonal": "add",
                "seasonal_periods": 12,
            }
        )

    rows: list[dict[str, Any]] = []
    for spec in candidate_specs:
        try:
            fit = fit_ets_model(
                train_series,
                trend=spec["trend"],
                seasonal=spec["seasonal"],
                seasonal_periods=spec["seasonal_periods"],
            )
            rows.append(
                {
                    **spec,
                    "aic": float(fit.aic),
                    "bic": float(fit.bic),
                }
            )
        except Exception:
            continue

    if not rows:
        raise DataTooShortError("Ingen ETS-kandidater kunne estimeres.")

    return pd.DataFrame(rows).sort_values(["aic", "bic"]).reset_index(drop=True)


def run_exponential_smoothing(
    train_panel: pd.DataFrame,
    test_panel: pd.DataFrame,
    split_metadata: dict[str, Any],
) -> tuple[ModelResult, pd.DataFrame]:
    predictions: list[dict[str, Any]] = []
    model_selection_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    representative_vessel = select_representative_vessel(train_panel)

    for vessel, test_vessel_df in test_panel.groupby("vessel"):
        train_series = build_vessel_series(train_panel, vessel).dropna()
        test_series = build_vessel_series(test_panel, vessel).dropna()
        if len(train_series) < 12 or test_series.empty:
            continue

        if train_series.nunique() <= 1:
            constant_value = float(train_series.iloc[-1])
            model_selection_rows.append(
                {
                    "vessel": vessel,
                    "spec": "CONST",
                    "aic": np.nan,
                    "bic": np.nan,
                    "train_observations": int(len(train_series)),
                }
            )
            residual_rows.append(
                {
                    "vessel": vessel,
                    "ljung_box_lag": np.nan,
                    "ljung_box_pvalue": np.nan,
                }
            )
            for date_value, actual in test_series.items():
                predictions.append(
                    {
                        "model": "exponential_smoothing",
                        "vessel": vessel,
                        "date": pd.Timestamp(date_value).strftime("%Y-%m-%d"),
                        "actual": float(actual),
                        "prediction": constant_value,
                    }
                )
            continue

        candidate_df = fit_ets_candidates(train_series)
        best_candidate = candidate_df.iloc[0]
        fit = fit_ets_model(
            train_series,
            trend=normalize_optional_string(best_candidate["trend"]),
            seasonal=normalize_optional_string(best_candidate["seasonal"]),
            seasonal_periods=(
                int(best_candidate["seasonal_periods"])
                if pd.notna(best_candidate["seasonal_periods"])
                else None
            ),
        )
        residuals = pd.Series(fit.resid, index=train_series.index).dropna()
        ljung_lag = min(12, max(len(residuals) // 2, 1))
        ljung_box = acorr_ljungbox(residuals, lags=[ljung_lag], return_df=True)

        model_selection_rows.append(
            {
                "vessel": vessel,
                "spec": best_candidate["spec"],
                "aic": float(best_candidate["aic"]),
                "bic": float(best_candidate["bic"]),
                "train_observations": int(len(train_series)),
            }
        )
        residual_rows.append(
            {
                "vessel": vessel,
                "ljung_box_lag": int(ljung_lag),
                "ljung_box_pvalue": float(ljung_box["lb_pvalue"].iloc[0]),
            }
        )

        history = train_series.copy()
        for date_value, actual in test_series.items():
            pred = float(
                fit_or_fallback_exponential_forecast(
                    history,
                    1,
                    trend=normalize_optional_string(best_candidate["trend"]),
                    seasonal=normalize_optional_string(best_candidate["seasonal"]),
                    seasonal_periods=(
                        int(best_candidate["seasonal_periods"])
                        if pd.notna(best_candidate["seasonal_periods"])
                        else None
                    ),
                )[0]
            )
            predictions.append(
                {
                    "model": "exponential_smoothing",
                    "vessel": vessel,
                    "date": pd.Timestamp(date_value).strftime("%Y-%m-%d"),
                    "actual": float(actual),
                    "prediction": pred,
                }
            )
            history = pd.concat(
                [history, pd.Series([float(actual)], index=pd.DatetimeIndex([date_value]))]
            )

    if not predictions:
        return (
            ModelResult(
                model="exponential_smoothing",
                status="skipped",
                details={
                    **split_metadata,
                    "reason": "For få observasjoner per fartøy til å kjøre eksponentiell glatting.",
                },
            ),
            pd.DataFrame(),
        )

    pred_df = pd.DataFrame(predictions)
    write_dataframe_artifacts(
        pd.DataFrame(model_selection_rows),
        model_artifact_path("exponential_smoothing", "modellvalg_per_fartoy.csv"),
        "Eksponentiell glatting modellvalg per fartøy",
    )
    write_dataframe_artifacts(
        pd.DataFrame(residual_rows),
        model_artifact_path("exponential_smoothing", "residualdiagnostikk.csv"),
        "Eksponentiell glatting residualdiagnostikk",
    )
    save_representative_prediction_plot(
        "exponential_smoothing",
        representative_vessel,
        train_panel,
        test_panel,
        pred_df,
        "Eksponentiell glatting: representativt testforløp",
    )
    mae_value, rmse_value, smape_value = summarize_prediction_frame(pred_df)
    metrics = ModelResult(
        model="exponential_smoothing",
        status="ok",
        mae=mae_value,
        rmse=rmse_value,
        smape=smape_value,
        details={
            **split_metadata,
            "vessels_used": int(pred_df["vessel"].nunique()),
            "test_rows": int(len(pred_df)),
            "evaluation_method": "ekspanderende 1-stegs prognose gjennom testperioden",
            "evaluation_level": "fartøynivå",
            "representative_vessel": representative_vessel,
            "selection_table": "modellvalg_per_fartoy.md",
            "residual_table": "residualdiagnostikk.md",
        },
    )
    return metrics, pred_df


def forecast_exponential_smoothing(
    panel_df: pd.DataFrame,
    horizon: int = MAX_FUTURE_FORECAST_HORIZON,
) -> pd.DataFrame:
    future_dates = future_dates_from_panel(panel_df, horizon)
    predictions: list[dict[str, Any]] = []

    for vessel, vessel_df in panel_df.groupby("vessel"):
        vessel_df = vessel_df.sort_values("date").reset_index(drop=True)
        if len(vessel_df) < 12:
            continue

        if vessel_df["offhire_days"].nunique() <= 1:
            forecast = np.repeat(float(vessel_df["offhire_days"].iloc[-1]), horizon)
            for forecast_step, (date_value, pred) in enumerate(
                zip(future_dates, forecast),
                start=1,
            ):
                predictions.append(
                    build_future_prediction_row(
                        "exponential_smoothing",
                        vessel,
                        date_value,
                        float(pred),
                        forecast_step,
                    )
                )
            continue

        candidate_df = fit_ets_candidates(vessel_df["offhire_days"].astype(float))
        best_candidate = candidate_df.iloc[0]
        forecast = fit_or_fallback_exponential_forecast(
            vessel_df["offhire_days"].astype(float),
            horizon,
            trend=normalize_optional_string(best_candidate["trend"]),
            seasonal=normalize_optional_string(best_candidate["seasonal"]),
            seasonal_periods=(
                int(best_candidate["seasonal_periods"])
                if pd.notna(best_candidate["seasonal_periods"])
                else None
            ),
        )
        for forecast_step, (date_value, pred) in enumerate(
            zip(future_dates, forecast),
            start=1,
        ):
            predictions.append(
                build_future_prediction_row(
                    "exponential_smoothing",
                    vessel,
                    date_value,
                    float(pred),
                    forecast_step,
                )
            )

    if not predictions:
        return pd.DataFrame()

    return pd.DataFrame(predictions).sort_values(["vessel", "date"]).reset_index(drop=True)
