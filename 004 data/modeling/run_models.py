from __future__ import annotations

import json

import pandas as pd

from common_config import ACTIVE_MODELS, GENERATE_FUTURE_FORECASTS, RESULTS_DIR
from common_data import (
    build_panel_features,
    build_split_metadata,
    ensure_split_datasets,
    format_period,
    load_dataset,
    validate_dataset_split,
)
from common_io import (
    cleanup_extra_artifacts,
    ensure_results_dir,
    save_future_forecast_artifacts,
    save_model_specific_outputs,
    save_summary_artifacts,
)
from common_types import DataTooShortError, ModelResult
from exponential_smoothing import (
    forecast_exponential_smoothing,
    run_exponential_smoothing,
)
from lstm_model import forecast_lstm, run_lstm
from sarima import forecast_sarima, run_sarima
from xgboost_model import forecast_xgboost, run_xgboost


def save_outputs(
    results: list[ModelResult],
    prediction_frames: dict[str, pd.DataFrame],
    future_prediction_frames: dict[str, pd.DataFrame],
    panel_df: pd.DataFrame,
    active_models: list[str],
    model_functions: dict[str, object],
) -> None:
    metrics_payload = [
        {
            "model": result.model,
            "status": result.status,
            "mae": result.mae,
            "rmse": result.rmse,
            "smape": result.smape,
            "details": result.details or {},
        }
        for result in results
    ]
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    save_model_specific_outputs(
        results,
        prediction_frames,
        future_prediction_frames,
        panel_df,
        active_models,
        model_functions=model_functions,
    )
    save_summary_artifacts(prediction_frames)

    non_empty_frames = [frame for frame in prediction_frames.values() if not frame.empty]
    if non_empty_frames:
        combined = pd.concat(
            non_empty_frames,
            ignore_index=True,
        )
        if not combined.empty:
            combined.to_csv(RESULTS_DIR / "predictions.csv", index=False)
    else:
        (RESULTS_DIR / "predictions.csv").unlink(missing_ok=True)

    save_future_forecast_artifacts(future_prediction_frames)


def main() -> None:
    ensure_results_dir()
    ensure_split_datasets()

    from common_config import MASTER_DATA_PATH, TEST_DATA_PATH, TRAIN_DATA_PATH

    full_panel = load_dataset(MASTER_DATA_PATH)
    train_panel = load_dataset(TRAIN_DATA_PATH)
    test_panel = load_dataset(TEST_DATA_PATH)
    validate_dataset_split(train_panel, test_panel, full_panel)
    evaluation_panel = (
        pd.concat([train_panel, test_panel], ignore_index=True)
        .sort_values(["vessel", "date"])
        .reset_index(drop=True)
    )
    split_metadata = build_split_metadata(train_panel, test_panel)
    active_models = ACTIVE_MODELS.copy()
    for model_name in active_models:
        cleanup_extra_artifacts(model_name)

    results: list[ModelResult] = []
    prediction_frames: dict[str, pd.DataFrame] = {}
    future_prediction_frames: dict[str, pd.DataFrame] = {
        model_name: pd.DataFrame() for model_name in active_models
    }
    model_functions = {
        "sarima": run_sarima,
        "exponential_smoothing": run_exponential_smoothing,
        "xgboost": run_xgboost,
        "lstm": run_lstm,
    }
    model_runners = {
        "sarima": lambda: run_sarima(train_panel, test_panel, split_metadata),
        "exponential_smoothing": lambda: run_exponential_smoothing(
            train_panel,
            test_panel,
            split_metadata,
        ),
        "xgboost": lambda: run_xgboost(evaluation_panel, split_metadata),
        "lstm": lambda: run_lstm(evaluation_panel, split_metadata),
    }

    for model_name in active_models:
        runner = model_runners[model_name]
        try:
            result, pred_df = runner()
            results.append(result)
            prediction_frames[model_name] = pred_df
        except DataTooShortError as exc:
            cleanup_extra_artifacts(model_name)
            results.append(
                ModelResult(
                    model=model_name,
                    status="skipped",
                    details={"reason": str(exc)},
                )
            )
        except Exception as exc:  # pragma: no cover - defensive logging for experimentation
            cleanup_extra_artifacts(model_name)
            results.append(
                ModelResult(
                    model=model_name,
                    status="failed",
                    details={"reason": str(exc)},
                )
            )

    if GENERATE_FUTURE_FORECASTS:
        forecast_features_df = build_panel_features(full_panel)
        future_forecasters = {
            "sarima": lambda details=None: forecast_sarima(full_panel, details),
            "exponential_smoothing": lambda: forecast_exponential_smoothing(full_panel),
            "xgboost": lambda: forecast_xgboost(full_panel, forecast_features_df),
            "lstm": lambda: forecast_lstm(full_panel),
        }

        for model_name in active_models:
            forecaster = future_forecasters[model_name]
            try:
                if model_name == "sarima":
                    sarima_result = next(
                        (result for result in results if result.model == "sarima"),
                        None,
                    )
                    future_prediction_frames[model_name] = forecaster(
                        sarima_result.details if sarima_result else None
                    )
                else:
                    future_prediction_frames[model_name] = forecaster()
            except DataTooShortError:
                future_prediction_frames[model_name] = pd.DataFrame()
            except Exception:
                future_prediction_frames[model_name] = pd.DataFrame()

    save_outputs(
        results,
        prediction_frames,
        future_prediction_frames,
        full_panel,
        active_models,
        model_functions,
    )

    print("Modellkjøring fullført.")
    print(
        "- evalueringssplit: "
        f"train {format_period(train_panel)} | test {format_period(test_panel)}"
    )
    for result in results:
        if result.status == "ok":
            print(
                f"- {result.model}: OK | MAE={result.mae:.3f} | RMSE={result.rmse:.3f}"
            )
        else:
            reason = (result.details or {}).get("reason", "ingen detalj")
            print(f"- {result.model}: {result.status.upper()} | {reason}")


if __name__ == "__main__":
    main()
