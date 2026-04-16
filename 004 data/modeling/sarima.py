from __future__ import annotations

from typing import Any

from common_config import MAX_FUTURE_FORECAST_HORIZON, MAX_OFFHIRE_VALUE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

from common_data import build_vessel_series, future_dates_from_panel
from common_eval import (
    build_future_prediction_row,
    select_representative_vessel,
    summarize_prediction_frame,
)
from common_io import (
    model_artifact_path,
    save_representative_prediction_plot,
    write_dataframe_artifacts,
)
from common_types import DataTooShortError, ModelResult


def fit_or_fallback_sarima_forecast(
    train_series: pd.Series,
    steps: int,
    order: tuple[int, int, int] = (1, 0, 1),
    seasonal_order: tuple[int, int, int, int] = (1, 0, 0, 12),
) -> np.ndarray:
    try:
        model = SARIMAX(
            train_series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = model.fit(disp=False)
        forecast = np.asarray(fit.forecast(steps=steps), dtype=float)
        if not np.isfinite(forecast).all() or float(np.nanmax(np.abs(forecast))) > 1000.0:
            raise ValueError("Forecast contains non-finite values.")
    except Exception:
        forecast = np.repeat(float(train_series.iloc[-1]), steps)

    return np.clip(forecast, 0.0, MAX_OFFHIRE_VALUE)


def run_adf_test(series: pd.Series, label: str, d: int, D: int) -> dict[str, Any]:
    clean_series = series.dropna().astype(float)
    if len(clean_series) < 12:
        return {
            "label": label,
            "d": d,
            "D": D,
            "n_obs": int(len(clean_series)),
            "adf_stat": None,
            "p_value": None,
            "stationary": False,
        }

    test_stat, p_value, _, _, critical_values, _ = adfuller(clean_series, autolag="AIC")
    return {
        "label": label,
        "d": d,
        "D": D,
        "n_obs": int(len(clean_series)),
        "adf_stat": float(test_stat),
        "p_value": float(p_value),
        "stationary": bool(p_value < 0.05),
        "critical_5pct": float(critical_values.get("5%", np.nan)),
    }


def apply_sarima_differencing(
    series: pd.Series,
    d: int,
    D: int,
    seasonal_period: int = 12,
) -> pd.Series:
    transformed = series.copy()
    for _ in range(d):
        transformed = transformed.diff()
    for _ in range(D):
        transformed = transformed.diff(seasonal_period)
    return transformed.dropna()


def select_sarima_differencing(train_series: pd.Series) -> tuple[int, int, pd.Series, list[dict[str, Any]]]:
    candidates = [
        {"label": "Ingen differensiering", "d": 0, "D": 0},
        {"label": "Første differense", "d": 1, "D": 0},
        {"label": "Sesongdifferense (12)", "d": 0, "D": 1},
        {"label": "Første + sesongdifferense", "d": 1, "D": 1},
    ]

    results: list[dict[str, Any]] = []
    for candidate in candidates:
        transformed = apply_sarima_differencing(
            train_series,
            d=candidate["d"],
            D=candidate["D"],
        )
        test_result = run_adf_test(
            transformed,
            label=candidate["label"],
            d=candidate["d"],
            D=candidate["D"],
        )
        results.append(test_result)

    stationary_candidates = [item for item in results if item["stationary"]]
    if stationary_candidates:
        selected = min(
            stationary_candidates,
            key=lambda item: (item["d"] + item["D"], item["p_value"] or float("inf")),
        )
    else:
        selected = min(results, key=lambda item: item["p_value"] or float("inf"))

    transformed_series = apply_sarima_differencing(
        train_series,
        d=int(selected["d"]),
        D=int(selected["D"]),
    )
    return int(selected["d"]), int(selected["D"]), transformed_series, results


def fit_sarima_candidates(
    train_series: pd.Series,
    d: int,
    D: int,
    seasonal_period: int = 12,
) -> pd.DataFrame:
    candidate_rows: list[dict[str, Any]] = []
    for p in range(0, 3):
        for q in range(0, 3):
            for seasonal_order in [(0, 0, 0, 0), *(
                [
                    (P, D, Q, seasonal_period)
                    for P in range(0, 2)
                    for Q in range(0, 2)
                    if not (P == 0 and Q == 0 and D == 0)
                ]
                if len(train_series) >= 36
                else []
            )]:
                try:
                    model = SARIMAX(
                        train_series,
                        order=(p, d, q),
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    fit = model.fit(disp=False)
                    candidate_rows.append(
                        {
                            "p": p,
                            "d": d,
                            "q": q,
                            "P": seasonal_order[0],
                            "D": seasonal_order[1],
                            "Q": seasonal_order[2],
                            "s": seasonal_order[3],
                            "aic": float(fit.aic),
                            "bic": float(fit.bic),
                            "complexity": p + q + seasonal_order[0] + seasonal_order[2] + d + seasonal_order[1],
                        }
                    )
                except Exception:
                    continue

    if not candidate_rows:
        raise DataTooShortError("Ingen SARIMA-kandidatmodeller kunne estimeres.")

    return (
        pd.DataFrame(candidate_rows)
        .sort_values(["aic", "bic", "complexity"])
        .reset_index(drop=True)
    )


def save_sarima_diagnostic_plots(
    representative_vessel: str,
    transformed_series: pd.Series,
    residuals: pd.Series,
) -> None:
    acf_path = model_artifact_path("sarima", "acf.png")
    pacf_path = model_artifact_path("sarima", "pacf.png")
    residuals_path = model_artifact_path("sarima", "residualdiagnostikk.png")
    acf_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    plot_acf(transformed_series.dropna(), ax=ax, lags=min(24, max(len(transformed_series.dropna()) - 1, 1)))
    ax.set_title(f"ARIMA/SARIMA ACF for {representative_vessel}")
    fig.tight_layout()
    fig.savefig(acf_path, dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    plot_pacf(
        transformed_series.dropna(),
        ax=ax,
        lags=min(24, max(len(transformed_series.dropna()) // 2 - 1, 1)),
        method="ywm",
    )
    ax.set_title(f"ARIMA/SARIMA PACF for {representative_vessel}")
    fig.tight_layout()
    fig.savefig(pacf_path, dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    axes[0].plot(residuals.index, residuals.values, color="#0F4C5C", linewidth=1.8)
    axes[0].axhline(0, color="#D97706", linestyle="--", linewidth=1)
    axes[0].set_title(f"ARIMA/SARIMA residualer over tid for {representative_vessel}")
    axes[0].set_ylabel("Residual")
    axes[1].hist(residuals.values, bins=12, color="#2C7A7B", alpha=0.85, edgecolor="white")
    axes[1].set_title(f"Fordeling av residualer for {representative_vessel}")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Frekvens")
    fig.tight_layout()
    fig.savefig(residuals_path, dpi=200)
    plt.close(fig)


def run_sarima(
    train_panel: pd.DataFrame,
    test_panel: pd.DataFrame,
    split_metadata: dict[str, Any],
) -> tuple[ModelResult, pd.DataFrame]:
    prediction_rows: list[dict[str, Any]] = []
    stationarity_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    representative_vessel = select_representative_vessel(train_panel)
    representative_candidates = pd.DataFrame()
    representative_transformed: pd.Series | None = None
    representative_residuals: pd.Series | None = None
    fallback_representative: str | None = None

    train_vessels = sorted(set(train_panel["vessel"]).intersection(test_panel["vessel"]))
    for vessel in train_vessels:
        train_series = build_vessel_series(train_panel, vessel).dropna()
        test_series = build_vessel_series(test_panel, vessel).dropna()
        if len(train_series) < 24 or test_series.empty:
            continue

        if train_series.nunique() <= 1:
            constant_value = float(train_series.iloc[-1])
            model_rows.append(
                {
                    "vessel": vessel,
                    "p": np.nan,
                    "d": np.nan,
                    "q": np.nan,
                    "P": np.nan,
                    "D": np.nan,
                    "Q": np.nan,
                    "s": 0,
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
                prediction_rows.append(
                    {
                        "model": "sarima",
                        "vessel": vessel,
                        "date": pd.Timestamp(date_value).strftime("%Y-%m-%d"),
                        "actual": float(actual),
                        "prediction": constant_value,
                    }
                )
            continue

        selected_d, selected_D, transformed_series, stationarity_results = select_sarima_differencing(
            train_series
        )
        stationarity_rows.extend(
            {"vessel": vessel, **row} for row in stationarity_results
        )
        candidate_df = fit_sarima_candidates(train_series, d=selected_d, D=selected_D)
        best_candidate = candidate_df.iloc[0]
        selected_order = (
            int(best_candidate["p"]),
            int(best_candidate["d"]),
            int(best_candidate["q"]),
        )
        selected_seasonal_order = (
            int(best_candidate["P"]),
            int(best_candidate["D"]),
            int(best_candidate["Q"]),
            int(best_candidate["s"]),
        )

        final_model = SARIMAX(
            train_series,
            order=selected_order,
            seasonal_order=selected_seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        final_fit = final_model.fit(disp=False)
        residuals = pd.Series(final_fit.resid, index=train_series.index).dropna()
        ljung_lag = min(12, max(len(residuals) // 2, 1))
        ljung_box = acorr_ljungbox(residuals, lags=[ljung_lag], return_df=True)

        model_rows.append(
            {
                "vessel": vessel,
                "p": selected_order[0],
                "d": selected_order[1],
                "q": selected_order[2],
                "P": selected_seasonal_order[0],
                "D": selected_seasonal_order[1],
                "Q": selected_seasonal_order[2],
                "s": selected_seasonal_order[3],
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
            prediction = float(
                fit_or_fallback_sarima_forecast(
                    history,
                    1,
                    order=selected_order,
                    seasonal_order=selected_seasonal_order,
                )[0]
            )
            prediction_rows.append(
                {
                    "model": "sarima",
                    "vessel": vessel,
                    "date": pd.Timestamp(date_value).strftime("%Y-%m-%d"),
                    "actual": float(actual),
                    "prediction": prediction,
                }
            )
            history = pd.concat(
                [history, pd.Series([float(actual)], index=pd.DatetimeIndex([date_value]))]
            )

        if fallback_representative is None:
            fallback_representative = vessel
        if representative_vessel == vessel or (
            representative_vessel not in train_vessels and fallback_representative == vessel
        ):
            representative_candidates = candidate_df.head(10).copy()
            representative_transformed = transformed_series
            representative_residuals = residuals
            representative_vessel = vessel

    if representative_candidates.empty and fallback_representative is not None:
        representative_vessel = fallback_representative

    if not prediction_rows:
        return (
            ModelResult(
                model="sarima",
                status="skipped",
                details={
                    **split_metadata,
                    "reason": "Ingen fartøy hadde tilstrekkelig historikk og variasjon til ARIMA/SARIMA.",
                },
            ),
            pd.DataFrame(),
        )

    pred_df = pd.DataFrame(prediction_rows)
    write_dataframe_artifacts(
        pd.DataFrame(stationarity_rows),
        model_artifact_path("sarima", "stasjonaritet.csv"),
        "ARIMA/SARIMA stasjonaritet per fartøy",
    )
    write_dataframe_artifacts(
        representative_candidates,
        model_artifact_path("sarima", "kandidatmodeller.csv"),
        "ARIMA/SARIMA kandidatmodeller for representativt fartøy",
    )
    write_dataframe_artifacts(
        pd.DataFrame(model_rows),
        model_artifact_path("sarima", "modellvalg_per_fartoy.csv"),
        "ARIMA/SARIMA modellvalg per fartøy",
    )
    write_dataframe_artifacts(
        pd.DataFrame(residual_rows),
        model_artifact_path("sarima", "residualdiagnostikk.csv"),
        "ARIMA/SARIMA residualdiagnostikk per fartøy",
    )
    if (
        representative_vessel is not None
        and representative_transformed is not None
        and representative_residuals is not None
    ):
        save_sarima_diagnostic_plots(
            representative_vessel,
            representative_transformed,
            representative_residuals,
        )
    save_representative_prediction_plot(
        "sarima",
        representative_vessel,
        train_panel,
        test_panel,
        pred_df,
        "ARIMA/SARIMA: representativt testforløp",
    )

    mae_value, rmse_value, smape_value = summarize_prediction_frame(pred_df)
    metrics = ModelResult(
        model="sarima",
        status="ok",
        mae=mae_value,
        rmse=rmse_value,
        smape=smape_value,
        details={
            **split_metadata,
            "series_type": "per_vessel",
            "test_rows": int(len(pred_df)),
            "evaluation_method": "ekspanderende 1-stegs prognose per fartøy",
            "evaluation_level": "fartøynivå",
            "modeled_vessels": int(pred_df["vessel"].nunique()),
            "representative_vessel": representative_vessel,
            "artifact_files": {
                "stasjonaritet": "stasjonaritet.md",
                "kandidatmodeller": "kandidatmodeller.md",
                "modellvalg_per_fartoy": "modellvalg_per_fartoy.md",
                "residualdiagnostikk_tabell": "residualdiagnostikk.md",
                "acf": "acf.png",
                "pacf": "pacf.png",
                "residualdiagnostikk_figur": "residualdiagnostikk.png",
                "representativ_testplot": "representativ_testplot.png",
            },
        },
    )
    return metrics, pred_df


def forecast_sarima(
    panel_df: pd.DataFrame,
    sarima_details: dict[str, Any] | None = None,
    horizon: int = MAX_FUTURE_FORECAST_HORIZON,
) -> pd.DataFrame:
    future_dates = future_dates_from_panel(panel_df, horizon)
    predictions: list[dict[str, Any]] = []
    for vessel in sorted(panel_df["vessel"].unique()):
        vessel_series = build_vessel_series(panel_df, vessel).dropna()
        if len(vessel_series) < 24:
            continue

        if vessel_series.nunique() <= 1:
            forecast = np.repeat(float(vessel_series.iloc[-1]), horizon)
            for forecast_step, (date_value, pred) in enumerate(
                zip(future_dates, forecast),
                start=1,
            ):
                predictions.append(
                    build_future_prediction_row(
                        "sarima",
                        vessel,
                        date_value,
                        float(pred),
                        forecast_step,
                    )
                )
            continue

        selected_d, selected_D, _, _ = select_sarima_differencing(vessel_series)
        candidate_df = fit_sarima_candidates(vessel_series, d=selected_d, D=selected_D)
        best_candidate = candidate_df.iloc[0]
        selected_order = (
            int(best_candidate["p"]),
            int(best_candidate["d"]),
            int(best_candidate["q"]),
        )
        selected_seasonal_order = (
            int(best_candidate["P"]),
            int(best_candidate["D"]),
            int(best_candidate["Q"]),
            int(best_candidate["s"]),
        )
        forecast = fit_or_fallback_sarima_forecast(
            vessel_series,
            horizon,
            order=selected_order,
            seasonal_order=selected_seasonal_order,
        )
        for forecast_step, (date_value, pred) in enumerate(
            zip(future_dates, forecast),
            start=1,
        ):
            predictions.append(
                build_future_prediction_row(
                    "sarima",
                    vessel,
                    date_value,
                    float(pred),
                    forecast_step,
                )
            )

    if not predictions:
        return pd.DataFrame()

    return pd.DataFrame(predictions).sort_values(["vessel", "date"]).reset_index(drop=True)
