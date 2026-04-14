from __future__ import annotations

import inspect
import json
import os
import re
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    module=r"statsmodels\.tsa\.holtwinters\.model",
)
warnings.filterwarnings(
    "ignore",
    category=ConvergenceWarning,
    module=r"statsmodels\.tsa\.holtwinters\.model",
)


ROOT = Path(__file__).resolve().parents[2]
MASTER_DATA_PATH = ROOT / "004 data" / "Data som skal brukes Anonymisert.csv"
TRAIN_DATA_PATH = ROOT / "004 data" / "train.csv"
TEST_DATA_PATH = ROOT / "004 data" / "test.csv"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
MODEL_LOG_DIR = Path(__file__).resolve().parent / "model_logs"
MODEL_ARTIFACTS_DIR = Path(__file__).resolve().parent / "models"
COMPARISON_LOG_PATH = MODEL_LOG_DIR / "Modellsammenligning.md"
TRAIN_END_DATE = pd.Timestamp(year=2024, month=12, day=1)
TEST_START_DATE = pd.Timestamp(year=2025, month=1, day=1)

MONTH_ORDER = {
    "Januar": 1,
    "Februar": 2,
    "Mars": 3,
    "April": 4,
    "Mai": 5,
    "Juni": 6,
    "Juli": 7,
    "August": 8,
    "September": 9,
    "Oktober": 10,
    "November": 11,
    "Desember": 12,
}
MONTH_COLUMNS = list(MONTH_ORDER.keys())
MONTH_NAME_BY_NUMBER = {value: key for key, value in MONTH_ORDER.items()}
YEAR_PATTERN = re.compile(r"År:\s*(\d{4})")

MODEL_METADATA = {
    "sarima": {
        "display_name": "SARIMA",
        "function_name": "run_sarima",
    },
    "exponential_smoothing": {
        "display_name": "Eksponentiell glatting",
        "function_name": "run_exponential_smoothing",
    },
    "xgboost": {
        "display_name": "XGBoost",
        "function_name": "run_xgboost",
    },
    "lstm": {
        "display_name": "LSTM",
        "function_name": "run_lstm",
    },
}
ACTIVE_MODELS = ["exponential_smoothing"]


@dataclass
class ModelResult:
    model: str
    status: str
    mae: float | None = None
    rmse: float | None = None
    details: dict[str, Any] | None = None


class DataTooShortError(RuntimeError):
    pass


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_LOG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def format_period(panel_df: pd.DataFrame) -> str:
    if panel_df.empty:
        return "ukjent"
    return f"{panel_df['date'].min():%Y-%m} til {panel_df['date'].max():%Y-%m}"


def build_split_metadata(
    train_panel: pd.DataFrame,
    test_panel: pd.DataFrame,
    forecast_panel: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "evaluation_train_period": format_period(train_panel),
        "evaluation_test_period": format_period(test_panel),
        "forecast_training_period": format_period(forecast_panel),
        "train_observations": int(len(train_panel)),
        "test_observations": int(len(test_panel)),
        "forecast_observations": int(len(forecast_panel)),
    }


def model_artifact_dir(model_name: str) -> Path:
    return MODEL_ARTIFACTS_DIR / MODEL_METADATA[model_name]["display_name"]


def model_artifact_path(model_name: str, filename: str) -> Path:
    return model_artifact_dir(model_name) / filename


def format_markdown_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value).replace("|", "\\|")


def dataframe_to_markdown(
    df: pd.DataFrame,
    max_rows: int = 10,
    empty_message: str = "_Ingen rader lagret._",
) -> str:
    if df.empty:
        return empty_message

    preview = df.head(max_rows).copy()
    headers = preview.columns.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in preview.iterrows():
        values = [format_markdown_value(row[column]) for column in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_model_code_log(model_name: str) -> None:
    metadata = MODEL_METADATA[model_name]
    function_name = metadata["function_name"]
    function_source = inspect.getsource(globals()[function_name]).rstrip()
    file_path = model_artifact_path(model_name, "kode.md")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        f"# {metadata['display_name']} kode\n\n"
        "Dette dokumentet genereres automatisk fra den aktive implementasjonen i "
        "`004 data/modeling/run_models.py`.\n\n"
        f"- Modell: `{model_name}`\n"
        f"- Funksjon: `{function_name}`\n\n"
        "```python\n"
        f"{function_source}\n"
        "```\n"
    )
    file_path.write_text(content, encoding="utf-8")


def write_model_result_log(
    result: ModelResult,
    pred_df: pd.DataFrame,
    future_df: pd.DataFrame,
    generated_at: str,
) -> None:
    metadata = MODEL_METADATA[result.model]
    file_path = model_artifact_path(result.model, "resultater.md")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    details_json = json.dumps(result.details or {}, indent=2, ensure_ascii=False)
    predictions_preview = dataframe_to_markdown(
        pred_df,
        empty_message="_Ingen evalueringsprediksjoner lagret for denne modellen._",
    )
    future_preview = dataframe_to_markdown(
        future_df,
        empty_message="_Ingen fremtidsprognoser lagret for denne modellen._",
    )
    details = result.details or {}
    content_lines = [
        f"# {metadata['display_name']} resultater",
        "",
        "Dette dokumentet genereres automatisk ved hver kjøring av "
        "`004 data/modeling/run_models.py`.",
        "",
        f"- Sist generert: `{generated_at}`",
        f"- Status: `{result.status}`",
        (
            f"- MAE: `{result.mae:.4f}`"
            if result.mae is not None
            else "- MAE: `ikke tilgjengelig`"
        ),
        (
            f"- RMSE: `{result.rmse:.4f}`"
            if result.rmse is not None
            else "- RMSE: `ikke tilgjengelig`"
        ),
    ]
    if "evaluation_train_period" in details and "evaluation_test_period" in details:
        content_lines.extend(
            [
                f"- Evaluering train: `{details['evaluation_train_period']}`",
                f"- Evaluering test: `{details['evaluation_test_period']}`",
            ]
        )
    if "forecast_training_period" in details:
        content_lines.append(
            f"- Fremtidsprognoser trener på full historikk: `{details['forecast_training_period']}`"
        )
    content_lines.extend(
        [
            "",
            "## Detaljer",
            "",
            "```json",
            details_json,
            "```",
            "",
            "## Prediksjoner",
            "",
            predictions_preview,
            "",
            "## Fremtidsprognoser",
            "",
            future_preview,
        ]
    )
    if not pred_df.empty and len(pred_df) > 10:
        content_lines.extend(
            [
                "",
                f"Viser de første 10 av totalt {len(pred_df)} prediksjonsrader.",
            ]
        )
    if not future_df.empty and len(future_df) > 10:
        content_lines.extend(
            [
                "",
                f"Viser de første 10 av totalt {len(future_df)} fremtidsprognoser.",
            ]
        )
    file_path.write_text("\n".join(content_lines) + "\n", encoding="utf-8")


def format_metric(value: float | None) -> str:
    if value is None:
        return "ikke tilgjengelig"
    return f"{value:.4f}"


def summarize_result_details(result: ModelResult) -> str:
    details = result.details or {}
    if result.status != "ok":
        return str(details.get("reason", "")).replace("|", "\\|")

    if result.model == "exponential_smoothing" and "vessels_used" in details:
        return f"Fartøy brukt: {details['vessels_used']}"
    if result.model == "xgboost":
        return (
            f"Train/Test-rader: {details.get('train_rows', 'ukjent')}/"
            f"{details.get('test_rows', 'ukjent')}"
        )
    if result.model == "lstm":
        return (
            f"Train/Test-sekvenser: {details.get('train_sequences', 'ukjent')}/"
            f"{details.get('test_sequences', 'ukjent')}"
        )
    if result.model == "sarima" and "series_type" in details:
        return f"Serie: {details['series_type']}"
    return ""


def build_dataset_notes(panel_df: pd.DataFrame) -> list[str]:
    if panel_df.empty:
        return ["- Ingen gyldige observasjoner ble lest fra datasettet."]

    min_date = panel_df["date"].min()
    max_date = panel_df["date"].max()
    unique_years = panel_df["date"].dt.year.nunique()
    notes = [
        (
            f"- Datasettet dekker observasjoner fra `{min_date:%Y-%m}` til "
            f"`{max_date:%Y-%m}` fordelt på {unique_years} kalenderår."
        )
    ]
    if max_date.month < 12:
        notes.append(
            (
                f"- Siste år i datasettet er foreløpig ufullstendig og går til "
                f"`{MONTH_NAME_BY_NUMBER[max_date.month]} {max_date.year}`."
            )
        )
    return notes


def future_dates_from_panel(panel_df: pd.DataFrame, horizon: int = 2) -> list[pd.Timestamp]:
    if panel_df.empty:
        raise DataTooShortError("Datasettet er tomt, så fremtidsdatoer kan ikke bygges.")

    start = panel_df["date"].max() + pd.offsets.MonthBegin(1)
    return list(pd.date_range(start=start, periods=horizon, freq="MS"))


def write_model_comparison_log(
    results: list[ModelResult],
    generated_at: str,
    panel_df: pd.DataFrame,
    future_prediction_frames: dict[str, pd.DataFrame],
    active_models: list[str],
) -> None:
    sorted_results = sorted(
        results,
        key=lambda result: (
            result.status != "ok",
            float("inf") if result.mae is None else result.mae,
            MODEL_METADATA[result.model]["display_name"],
        ),
    )
    successful_results = [
        result
        for result in sorted_results
        if result.status == "ok" and result.mae is not None and result.rmse is not None
    ]
    skipped_or_failed = [
        result for result in sorted_results if result.status in {"skipped", "failed"}
    ]

    content_lines = [
        "# Modellsammenligning",
        "",
        "Dette dokumentet oppsummerer siste kjøring av alle modellene og er ment som "
        "arbeidsgrunnlag for metode-, analyse- og diskusjonsdelen i rapporten.",
        "",
        f"- Sist generert: `{generated_at}`",
    ]
    reference_details = next(
        (
            result.details
            for result in sorted_results
            if result.details and "evaluation_train_period" in result.details
        ),
        None,
    )
    if reference_details:
        content_lines.extend(
            [
                f"- Evaluering train: `{reference_details['evaluation_train_period']}`",
                f"- Evaluering test: `{reference_details['evaluation_test_period']}`",
                f"- Fremtidsprognoser trener på full historikk: `{reference_details['forecast_training_period']}`",
            ]
        )
    content_lines.extend(
        [
            "",
            "## Samlet oversikt",
            "",
            "| Modell | Status | MAE | RMSE | Kommentar |",
            "| --- | --- | --- | --- | --- |",
        ]
    )

    for result in sorted_results:
        display_name = MODEL_METADATA[result.model]["display_name"]
        detail_summary = summarize_result_details(result) or "-"
        content_lines.append(
            "| "
            + " | ".join(
                [
                    display_name,
                    result.status,
                    format_metric(result.mae),
                    format_metric(result.rmse),
                    detail_summary,
                ]
            )
            + " |"
        )

    content_lines.extend(
        [
            "",
            "## Lenker til detaljfiler",
            "",
        ]
    )
    for model_name in active_models:
        display_name = MODEL_METADATA[model_name]["display_name"]
        content_lines.append(
            f"- [{display_name} kode](../models/{display_name}/kode.md) og "
            f"[{display_name} resultater](../models/{display_name}/resultater.md)"
        )

    content_lines.extend(
        [
            "",
            "## Rangering basert på MAE",
            "",
        ]
    )
    if successful_results:
        for index, result in enumerate(successful_results, start=1):
            display_name = MODEL_METADATA[result.model]["display_name"]
            content_lines.append(
                f"{index}. {display_name} med MAE {result.mae:.4f} og RMSE {result.rmse:.4f}"
            )
    else:
        content_lines.append("Ingen modeller med fullførte resultater i siste kjøring.")

    content_lines.extend(
        [
            "",
            "## Hovedfunn",
            "",
        ]
    )
    if successful_results:
        best_mae = min(successful_results, key=lambda result: result.mae or float("inf"))
        best_rmse = min(successful_results, key=lambda result: result.rmse or float("inf"))
        content_lines.append(
            f"- Lavest MAE i siste kjøring: `{MODEL_METADATA[best_mae.model]['display_name']}` "
            f"({best_mae.mae:.4f})."
        )
        content_lines.append(
            f"- Lavest RMSE i siste kjøring: `{MODEL_METADATA[best_rmse.model]['display_name']}` "
            f"({best_rmse.rmse:.4f})."
        )
    else:
        content_lines.append("- Ingen modeller produserte komplette metrikker i siste kjøring.")

    if skipped_or_failed:
        for result in skipped_or_failed:
            display_name = MODEL_METADATA[result.model]["display_name"]
            reason = (result.details or {}).get("reason", "Ingen detalj oppgitt.")
            content_lines.append(f"- `{display_name}` ble {result.status} i siste kjøring: {reason}")

    future_frames = [frame for frame in future_prediction_frames.values() if not frame.empty]
    if future_frames:
        combined_future = pd.concat(future_frames, ignore_index=True)
        unique_future_dates = sorted(pd.to_datetime(combined_future["date"]).dt.strftime("%Y-%m-%d").unique())
        content_lines.extend(
            [
                "",
                "## Fremtidsprognoser",
                "",
                (
                    "- Prognoser for neste måneder er lagret i "
                    "`004 data/modeling/results/future_predictions.csv` for datoene "
                    + ", ".join(f"`{date}`" for date in unique_future_dates)
                    + "."
                ),
            ]
        )

    content_lines.extend(
        [
            *build_dataset_notes(panel_df),
            "",
            "## Notater Til Rapport",
            "",
            "- Bruk tabellen over direkte som grunnlag for sammenligning av modellprestasjon.",
            "- Beskriv eksplisitt hvilken tidsdekning datasettet faktisk har, og at siste år kan være ufullstendig.",
            "- Diskuter om lav MAE alene er nok, eller om modellens tolkbarhet og datakrav også bør vektlegges.",
        ]
    )

    COMPARISON_LOG_PATH.write_text("\n".join(content_lines) + "\n", encoding="utf-8")


def save_model_specific_outputs(
    results: list[ModelResult],
    prediction_frames: dict[str, pd.DataFrame],
    future_prediction_frames: dict[str, pd.DataFrame],
    panel_df: pd.DataFrame,
    active_models: list[str],
) -> None:
    generated_at = datetime.now().isoformat(timespec="seconds")
    for model_name in active_models:
        write_model_code_log(model_name)

        metrics_path = model_artifact_path(model_name, "metrics.json")
        predictions_path = model_artifact_path(model_name, "predictions.csv")
        future_predictions_path = model_artifact_path(model_name, "future_predictions.csv")

        result = next((item for item in results if item.model == model_name), None)
        pred_df = prediction_frames.get(model_name, pd.DataFrame())
        future_df = future_prediction_frames.get(model_name, pd.DataFrame())

        if result is None:
            write_model_result_log(
                ModelResult(model=model_name, status="not_run", details={}),
                pred_df,
                future_df,
                generated_at,
            )
            metrics_path.unlink(missing_ok=True)
            predictions_path.unlink(missing_ok=True)
            future_predictions_path.unlink(missing_ok=True)
            continue

        metrics_payload = {
            "model": result.model,
            "status": result.status,
            "mae": result.mae,
            "rmse": result.rmse,
            "details": result.details or {},
        }
        metrics_path.write_text(
            json.dumps(metrics_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        if pred_df.empty:
            predictions_path.unlink(missing_ok=True)
        else:
            pred_df.to_csv(predictions_path, index=False)

        if future_df.empty:
            future_predictions_path.unlink(missing_ok=True)
        else:
            future_df.to_csv(future_predictions_path, index=False)

        write_model_result_log(result, pred_df, future_df, generated_at)

    write_model_comparison_log(
        results,
        generated_at,
        panel_df,
        future_prediction_frames,
        active_models,
    )


def load_dataset(data_path: Path) -> pd.DataFrame:
    raw_df = pd.read_csv(
        data_path,
        sep=";",
        header=None,
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
    )
    records: list[dict[str, Any]] = []
    current_year: int | None = None

    for row in raw_df.itertuples(index=False, name=None):
        values = ["" if pd.isna(value) else str(value).strip() for value in row]
        if not any(values):
            continue

        label = values[1] if len(values) > 1 else ""
        year_match = YEAR_PATTERN.fullmatch(label)
        if year_match:
            current_year = int(year_match.group(1))
            continue

        if label == "Måned/Fartøy":
            continue

        if not label.startswith("Fartøy ") or current_year is None:
            continue

        special_requirements = values[14] if len(values) > 14 else ""
        for column_index, month_name in enumerate(MONTH_COLUMNS, start=2):
            raw_value = values[column_index] if len(values) > column_index else ""
            if raw_value in {"", "N/A"}:
                continue

            numeric_value = (
                raw_value.replace("%", "").replace(" ", "").replace(",", ".")
            )
            if numeric_value == "":
                continue

            records.append(
                {
                    "vessel": re.sub(r"\s+", " ", label).strip(),
                    "month_name": month_name,
                    "month_num": MONTH_ORDER[month_name],
                    "date": pd.Timestamp(
                        year=current_year,
                        month=MONTH_ORDER[month_name],
                        day=1,
                    ),
                    "offhire_days": float(numeric_value),
                    "Spesielle behov/krav": special_requirements,
                }
            )

    if not records:
        raise DataTooShortError("Fant ingen gyldige observasjoner i CSV-filen.")

    long_df = pd.DataFrame.from_records(records)
    long_df = long_df.sort_values(["vessel", "date"]).reset_index(drop=True)
    return long_df[
        ["vessel", "month_name", "month_num", "date", "offhire_days", "Spesielle behov/krav"]
    ]


def write_split_dataset(
    master_path: Path,
    output_path: Path,
    years_to_keep: set[int],
) -> None:
    raw_df = pd.read_csv(
        master_path,
        sep=";",
        header=None,
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
    )
    year_row_indexes = [
        index
        for index, value in raw_df.iloc[:, 1].items()
        if YEAR_PATTERN.fullmatch(str(value).strip())
    ]
    if not year_row_indexes:
        raise DataTooShortError("Fant ingen årsblokker i råfilen som kunne splittes.")

    selected_blocks: list[pd.DataFrame] = []
    for position, start_idx in enumerate(year_row_indexes):
        year_match = YEAR_PATTERN.fullmatch(str(raw_df.iat[start_idx, 1]).strip())
        if year_match is None:
            continue
        year_value = int(year_match.group(1))
        end_idx = year_row_indexes[position + 1] if position + 1 < len(year_row_indexes) else len(raw_df)
        if year_value not in years_to_keep:
            continue

        block = raw_df.iloc[start_idx:end_idx].copy()
        if not selected_blocks:
            block.iat[0, 0] = str(raw_df.iat[0, 0]).strip()
        selected_blocks.append(block)

    if not selected_blocks:
        raise DataTooShortError(
            f"Fant ingen årsblokker for {output_path.name} i råfilen."
        )

    split_df = pd.concat(selected_blocks, ignore_index=True)
    split_df.to_csv(output_path, sep=";", index=False, header=False, encoding="utf-8-sig")


def ensure_split_datasets() -> None:
    write_split_dataset(MASTER_DATA_PATH, TRAIN_DATA_PATH, {2021, 2022, 2023, 2024})
    write_split_dataset(MASTER_DATA_PATH, TEST_DATA_PATH, {2025, 2026})


def validate_dataset_split(
    train_panel: pd.DataFrame,
    test_panel: pd.DataFrame,
    forecast_panel: pd.DataFrame,
) -> None:
    if train_panel.empty or test_panel.empty:
        raise DataTooShortError("Train/test-splittet ble tomt.")
    if train_panel["date"].max() > TRAIN_END_DATE:
        raise DataTooShortError("Train-datasettet inneholder datoer etter 2024-12.")
    if test_panel["date"].min() < TEST_START_DATE:
        raise DataTooShortError("Test-datasettet inneholder datoer før 2025-01.")
    if train_panel["date"].max() >= test_panel["date"].min():
        raise DataTooShortError("Train- og testperiodene overlapper.")

    combined = (
        pd.concat([train_panel, test_panel], ignore_index=True)
        .sort_values(["vessel", "date"])
        .reset_index(drop=True)
    )
    forecast_sorted = forecast_panel.sort_values(["vessel", "date"]).reset_index(drop=True)

    comparison_columns = [
        "vessel",
        "month_name",
        "month_num",
        "date",
        "offhire_days",
        "Spesielle behov/krav",
    ]
    if not combined[comparison_columns].equals(forecast_sorted[comparison_columns]):
        raise DataTooShortError(
            "Train/test-splittet matcher ikke hele masterdatasettet etter sammenslåing."
        )


def build_fleet_series(panel_df: pd.DataFrame) -> pd.Series:
    fleet_series = panel_df.groupby("date", as_index=True)["offhire_days"].sum().sort_index()
    fleet_series.index = pd.DatetimeIndex(fleet_series.index)
    return fleet_series.asfreq("MS")


def build_panel_features(panel_df: pd.DataFrame, lags: int = 3) -> pd.DataFrame:
    df = panel_df.copy()
    df["special_flag"] = df["Spesielle behov/krav"].fillna("").str.strip().ne("")

    for lag in range(1, lags + 1):
        df[f"lag_{lag}"] = df.groupby("vessel")["offhire_days"].shift(lag)

    df["rolling_mean_3"] = df.groupby("vessel")["offhire_days"].transform(
        lambda series: series.shift(1).rolling(window=3, min_periods=1).mean()
    )
    df["rolling_std_3"] = (
        df.groupby("vessel")["offhire_days"].transform(
            lambda series: series.shift(1).rolling(window=3, min_periods=1).std()
        )
    ).fillna(0.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12.0)

    df = df.dropna(subset=[f"lag_{lags}"]).reset_index(drop=True)
    return df


def split_features_for_evaluation(
    features_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = features_df[features_df["date"] <= TRAIN_END_DATE].copy()
    test_df = features_df[features_df["date"] >= TEST_START_DATE].copy()
    if train_df.empty or test_df.empty:
        raise DataTooShortError("Train/test-splitt ble tom.")
    return train_df, test_df


def fit_or_fallback_exponential_forecast(
    train_series: pd.Series, steps: int
) -> np.ndarray:
    trend = "add" if train_series.nunique() > 1 and len(train_series) >= 5 else None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            warnings.simplefilter("ignore", category=RuntimeWarning)
            model = ExponentialSmoothing(
                train_series,
                trend=trend,
                seasonal=None,
                initialization_method="estimated",
            )
            fit = model.fit(optimized=True, use_brute=True)
        forecast = np.asarray(fit.forecast(steps), dtype=float)
        if not np.isfinite(forecast).all():
            raise ValueError("Forecast contains non-finite values.")
    except Exception:
        forecast = np.repeat(float(train_series.iloc[-1]), steps)

    return np.clip(forecast, 0.0, None)


def fit_or_fallback_sarima_forecast(train_series: pd.Series, steps: int) -> np.ndarray:
    try:
        model = SARIMAX(
            train_series,
            order=(1, 0, 1),
            seasonal_order=(1, 0, 0, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = model.fit(disp=False)
        forecast = np.asarray(fit.forecast(steps=steps), dtype=float)
        if not np.isfinite(forecast).all():
            raise ValueError("Forecast contains non-finite values.")
    except Exception:
        forecast = np.repeat(float(train_series.iloc[-1]), steps)

    return np.clip(forecast, 0.0, None)


def run_exponential_smoothing(
    train_panel: pd.DataFrame,
    test_panel: pd.DataFrame,
    split_metadata: dict[str, Any],
) -> tuple[ModelResult, pd.DataFrame]:
    predictions: list[dict[str, Any]] = []
    for vessel, test_vessel_df in test_panel.groupby("vessel"):
        train_vessel_df = (
            train_panel[train_panel["vessel"] == vessel]
            .sort_values("date")
            .reset_index(drop=True)
        )
        test_vessel_df = test_vessel_df.sort_values("date").reset_index(drop=True)
        if len(train_vessel_df) < 4 or test_vessel_df.empty:
            continue

        history_values = train_vessel_df["offhire_days"].astype(float).tolist()

        for _, test_row in test_vessel_df.iterrows():
            train_series = pd.Series(history_values, dtype=float)
            pred = float(fit_or_fallback_exponential_forecast(train_series, 1)[0])
            predictions.append(
                {
                    "model": "exponential_smoothing",
                    "vessel": vessel,
                    "date": test_row["date"].strftime("%Y-%m-%d"),
                    "actual": float(test_row["offhire_days"]),
                    "prediction": pred,
                }
            )
            history_values.append(float(test_row["offhire_days"]))

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
    metrics = ModelResult(
        model="exponential_smoothing",
        status="ok",
        mae=float(mean_absolute_error(pred_df["actual"], pred_df["prediction"])),
        rmse=rmse(pred_df["actual"].to_numpy(), pred_df["prediction"].to_numpy()),
        details={
            **split_metadata,
            "vessels_used": int(pred_df["vessel"].nunique()),
            "test_rows": int(len(pred_df)),
            "evaluation_method": "ekspanderende 1-stegs prognose gjennom testperioden",
        },
    )
    return metrics, pred_df


def forecast_exponential_smoothing(panel_df: pd.DataFrame, horizon: int = 2) -> pd.DataFrame:
    future_dates = future_dates_from_panel(panel_df, horizon)
    predictions: list[dict[str, Any]] = []

    for vessel, vessel_df in panel_df.groupby("vessel"):
        vessel_df = vessel_df.sort_values("date").reset_index(drop=True)
        if len(vessel_df) < 4:
            continue

        forecast = fit_or_fallback_exponential_forecast(
            vessel_df["offhire_days"].astype(float), horizon
        )
        for date_value, pred in zip(future_dates, forecast):
            predictions.append(
                {
                    "model": "exponential_smoothing",
                    "vessel": vessel,
                    "date": date_value.strftime("%Y-%m-%d"),
                    "prediction": float(pred),
                }
            )

    if not predictions:
        return pd.DataFrame()

    return pd.DataFrame(predictions).sort_values(["vessel", "date"]).reset_index(drop=True)


def run_sarima(
    train_panel: pd.DataFrame,
    test_panel: pd.DataFrame,
    split_metadata: dict[str, Any],
) -> tuple[ModelResult, pd.DataFrame]:
    train_series = build_fleet_series(train_panel)
    test_series = build_fleet_series(test_panel)
    if len(train_series) < 24:
        return (
            ModelResult(
                model="sarima",
                status="skipped",
                details={
                    **split_metadata,
                    "reason": (
                        "SARIMA ble ikke kjørt fordi dataserien er for kort for en forsvarlig "
                        "månedlig sesongmodell. Minst 24 observasjoner anbefales."
                    ),
                    "observations": int(len(train_series)),
                },
            ),
            pd.DataFrame(),
        )

    history = train_series.copy()
    prediction_rows: list[dict[str, Any]] = []
    for date_value, actual in test_series.items():
        prediction = float(fit_or_fallback_sarima_forecast(history, 1)[0])
        prediction_rows.append(
            {
                "model": "sarima",
                "vessel": "fleet_total",
                "date": pd.Timestamp(date_value).strftime("%Y-%m-%d"),
                "actual": float(actual),
                "prediction": prediction,
            }
        )
        history = pd.concat(
            [history, pd.Series([float(actual)], index=pd.DatetimeIndex([date_value]))]
        )

    pred_df = pd.DataFrame(
        prediction_rows
    )
    metrics = ModelResult(
        model="sarima",
        status="ok",
        mae=float(mean_absolute_error(pred_df["actual"], pred_df["prediction"])),
        rmse=rmse(pred_df["actual"].to_numpy(), pred_df["prediction"].to_numpy()),
        details={
            **split_metadata,
            "series_type": "fleet_total",
            "test_rows": int(len(pred_df)),
            "evaluation_method": "ekspanderende 1-stegs prognose på aggregert flåteserie",
        },
    )
    return metrics, pred_df


def forecast_sarima(panel_df: pd.DataFrame, horizon: int = 2) -> pd.DataFrame:
    fleet_series = build_fleet_series(panel_df)
    if len(fleet_series) < 24:
        return pd.DataFrame()

    future_dates = future_dates_from_panel(panel_df, horizon)
    model = SARIMAX(
        fleet_series,
        order=(1, 0, 1),
        seasonal_order=(1, 0, 0, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)
    forecast = np.clip(np.asarray(fit.forecast(steps=horizon), dtype=float), 0.0, None)
    return pd.DataFrame(
        {
            "model": "sarima",
            "vessel": "fleet_total",
            "date": [date_value.strftime("%Y-%m-%d") for date_value in future_dates],
            "prediction": forecast,
        }
    )


def build_xgboost_pipeline() -> tuple[Pipeline, list[str]]:
    from xgboost import XGBRegressor

    numeric_features = [
        "month_num",
        "lag_1",
        "lag_2",
        "lag_3",
        "rolling_mean_3",
        "rolling_std_3",
        "month_sin",
        "month_cos",
    ]
    categorical_features = ["vessel", "special_flag"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ]
    )

    model = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    return pipeline, numeric_features + categorical_features


def run_xgboost(
    features_df: pd.DataFrame,
    split_metadata: dict[str, Any],
) -> tuple[ModelResult, pd.DataFrame]:
    train_df, test_df = split_features_for_evaluation(features_df)
    pipeline, feature_columns = build_xgboost_pipeline()
    pipeline.fit(train_df[feature_columns], train_df["offhire_days"])
    predictions = np.clip(pipeline.predict(test_df[feature_columns]), 0.0, None)

    pred_df = test_df[["vessel", "date", "offhire_days"]].copy()
    pred_df["model"] = "xgboost"
    pred_df["actual"] = pred_df["offhire_days"].astype(float)
    pred_df["prediction"] = predictions.astype(float)
    pred_df = pred_df.drop(columns=["offhire_days"])
    pred_df["date"] = pred_df["date"].dt.strftime("%Y-%m-%d")

    metrics = ModelResult(
        model="xgboost",
        status="ok",
        mae=float(mean_absolute_error(pred_df["actual"], pred_df["prediction"])),
        rmse=rmse(pred_df["actual"].to_numpy(), pred_df["prediction"].to_numpy()),
        details={
            **split_metadata,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "evaluation_method": "fast holdout med historiske lags og testperiode 2025-01 til 2026-03",
        },
    )
    return metrics, pred_df


def forecast_xgboost(
    panel_df: pd.DataFrame,
    features_df: pd.DataFrame,
    horizon: int = 2,
) -> pd.DataFrame:
    if features_df.empty:
        return pd.DataFrame()

    pipeline, feature_columns = build_xgboost_pipeline()
    pipeline.fit(features_df[feature_columns], features_df["offhire_days"])
    future_dates = future_dates_from_panel(panel_df, horizon)
    predictions: list[dict[str, Any]] = []

    for vessel, vessel_df in panel_df.groupby("vessel"):
        vessel_df = vessel_df.sort_values("date").reset_index(drop=True)
        if len(vessel_df) < 3:
            continue

        history_values = vessel_df["offhire_days"].astype(float).tolist()
        special_flag = bool(
            vessel_df["Spesielle behov/krav"].fillna("").astype(str).str.strip().iloc[-1]
        )

        for date_value in future_dates:
            latest_values = np.asarray(history_values[-3:], dtype=float)
            feature_row = pd.DataFrame(
                [
                    {
                        "month_num": date_value.month,
                        "lag_1": float(latest_values[-1]),
                        "lag_2": float(latest_values[-2]),
                        "lag_3": float(latest_values[-3]),
                        "rolling_mean_3": float(latest_values.mean()),
                        "rolling_std_3": (
                            float(np.std(latest_values, ddof=1))
                            if len(latest_values) > 1
                            else 0.0
                        ),
                        "month_sin": float(np.sin(2 * np.pi * date_value.month / 12.0)),
                        "month_cos": float(np.cos(2 * np.pi * date_value.month / 12.0)),
                        "vessel": vessel,
                        "special_flag": special_flag,
                    }
                ]
            )
            prediction = float(
                np.clip(pipeline.predict(feature_row[feature_columns])[0], 0.0, None)
            )
            predictions.append(
                {
                    "model": "xgboost",
                    "vessel": vessel,
                    "date": date_value.strftime("%Y-%m-%d"),
                    "prediction": prediction,
                }
            )
            history_values.append(prediction)

    if not predictions:
        return pd.DataFrame()

    return pd.DataFrame(predictions).sort_values(["vessel", "date"]).reset_index(drop=True)


def build_lstm_sequences(
    panel_df: pd.DataFrame, window_size: int = 3
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
) -> tuple[Any, StandardScaler, StandardScaler]:
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
        "epochs": 200,
        "batch_size": 4,
        "verbose": 0,
        "shuffle": False,
    }
    if len(X_train_scaled) >= 10:
        fit_kwargs["validation_split"] = 0.2
        fit_kwargs["callbacks"] = [callback]

    model.fit(**fit_kwargs)
    return model, x_scaler, y_scaler


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
        },
    )
    return metrics, pred_df


def forecast_lstm(panel_df: pd.DataFrame, horizon: int = 2) -> pd.DataFrame:
    X, y, _ = build_lstm_sequences(panel_df, window_size=3)
    model, x_scaler, y_scaler = train_lstm_regressor(X, y)
    future_dates = future_dates_from_panel(panel_df, horizon)
    predictions: list[dict[str, Any]] = []

    for vessel, vessel_df in panel_df.groupby("vessel"):
        vessel_df = vessel_df.sort_values("date").reset_index(drop=True)
        if len(vessel_df) <= 3:
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

        for date_value in future_dates:
            sequence = np.stack(
                [
                    np.asarray(history_values[-3:], dtype=np.float32),
                    np.sin(2 * np.pi * np.asarray(history_months[-3:], dtype=np.float32) / 12.0),
                    np.cos(2 * np.pi * np.asarray(history_months[-3:], dtype=np.float32) / 12.0),
                    np.asarray(history_special[-3:], dtype=np.float32),
                ],
                axis=1,
            )
            sequence_scaled = x_scaler.transform(sequence).reshape(1, sequence.shape[0], sequence.shape[1])
            prediction_scaled = model.predict(sequence_scaled, verbose=0).reshape(-1, 1)
            prediction = float(
                np.clip(y_scaler.inverse_transform(prediction_scaled).reshape(-1)[0], 0.0, None)
            )
            predictions.append(
                {
                    "model": "lstm",
                    "vessel": vessel,
                    "date": date_value.strftime("%Y-%m-%d"),
                    "prediction": prediction,
                }
            )
            history_values.append(prediction)
            history_months.append(float(date_value.month))
            history_special.append(float(latest_special_flag))

    if not predictions:
        return pd.DataFrame()

    return pd.DataFrame(predictions).sort_values(["vessel", "date"]).reset_index(drop=True)


def save_outputs(
    results: list[ModelResult],
    prediction_frames: dict[str, pd.DataFrame],
    future_prediction_frames: dict[str, pd.DataFrame],
    panel_df: pd.DataFrame,
    active_models: list[str],
) -> None:
    metrics_payload = [
        {
            "model": result.model,
            "status": result.status,
            "mae": result.mae,
            "rmse": result.rmse,
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
    )

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

    non_empty_future_frames = [
        frame for frame in future_prediction_frames.values() if not frame.empty
    ]
    if non_empty_future_frames:
        combined_future = pd.concat(non_empty_future_frames, ignore_index=True)
        combined_future = combined_future.sort_values(["model", "vessel", "date"]).reset_index(drop=True)
        combined_future.to_csv(RESULTS_DIR / "future_predictions.csv", index=False)
    else:
        (RESULTS_DIR / "future_predictions.csv").unlink(missing_ok=True)


def main() -> None:
    ensure_results_dir()
    ensure_split_datasets()
    full_panel = load_dataset(MASTER_DATA_PATH)
    train_panel = load_dataset(TRAIN_DATA_PATH)
    test_panel = load_dataset(TEST_DATA_PATH)
    validate_dataset_split(train_panel, test_panel, full_panel)
    evaluation_panel = (
        pd.concat([train_panel, test_panel], ignore_index=True)
        .sort_values(["vessel", "date"])
        .reset_index(drop=True)
    )
    split_metadata = build_split_metadata(train_panel, test_panel, full_panel)
    active_models = ACTIVE_MODELS.copy()

    results: list[ModelResult] = []
    prediction_frames: dict[str, pd.DataFrame] = {}
    future_prediction_frames: dict[str, pd.DataFrame] = {}

    evaluation_features_df = build_panel_features(evaluation_panel, lags=3)
    forecast_features_df = build_panel_features(full_panel, lags=3)

    model_runners = {
        "sarima": lambda: run_sarima(train_panel, test_panel, split_metadata),
        "exponential_smoothing": lambda: run_exponential_smoothing(
            train_panel,
            test_panel,
            split_metadata,
        ),
        "xgboost": lambda: run_xgboost(evaluation_features_df, split_metadata),
        "lstm": lambda: run_lstm(evaluation_panel, split_metadata),
    }
    future_forecasters = {
        "sarima": lambda: forecast_sarima(full_panel),
        "exponential_smoothing": lambda: forecast_exponential_smoothing(full_panel),
        "xgboost": lambda: forecast_xgboost(full_panel, forecast_features_df),
        "lstm": lambda: forecast_lstm(full_panel),
    }

    for model_name in active_models:
        runner = model_runners[model_name]
        try:
            result, pred_df = runner()
            results.append(result)
            prediction_frames[model_name] = pred_df
        except DataTooShortError as exc:
            results.append(
                ModelResult(
                    model=model_name,
                    status="skipped",
                    details={"reason": str(exc)},
                )
            )
        except Exception as exc:  # pragma: no cover - defensive logging for experimentation
            results.append(
                ModelResult(
                    model=model_name,
                    status="failed",
                    details={"reason": str(exc)},
                )
            )

    for model_name in active_models:
        forecaster = future_forecasters[model_name]
        try:
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

    combined_future_rows = sum(len(frame) for frame in future_prediction_frames.values())
    if combined_future_rows:
        future_dates = sorted(
            {
                date_value
                for frame in future_prediction_frames.values()
                if not frame.empty
                for date_value in frame["date"].tolist()
            }
        )
        print(
            "- fremtidsprognoser: "
            f"{combined_future_rows} rader for {', '.join(future_dates)}"
        )


if __name__ == "__main__":
    main()
