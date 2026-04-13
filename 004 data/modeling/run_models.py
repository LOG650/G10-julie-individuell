from __future__ import annotations

import inspect
import json
import os
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
DATA_PATH = ROOT / "004 data" / "Data som skal brukes Anonymisert.csv"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
MODEL_LOG_DIR = Path(__file__).resolve().parent / "model_logs"
COMPARISON_LOG_PATH = MODEL_LOG_DIR / "Modellsammenligning.md"

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
SYNTHETIC_YEAR = 2025

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


def format_markdown_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value).replace("|", "\\|")


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 10) -> str:
    if df.empty:
        return "_Ingen prediksjoner lagret for denne modellen._"

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
    file_path = MODEL_LOG_DIR / f"{metadata['display_name']} kode.md"
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
    generated_at: str,
) -> None:
    metadata = MODEL_METADATA[result.model]
    file_path = MODEL_LOG_DIR / f"{metadata['display_name']} resultater.md"
    details_json = json.dumps(result.details or {}, indent=2, ensure_ascii=False)
    predictions_preview = dataframe_to_markdown(pred_df)
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
    ]
    if not pred_df.empty and len(pred_df) > 10:
        content_lines.extend(
            [
                "",
                f"Viser de første 10 av totalt {len(pred_df)} prediksjonsrader.",
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


def write_model_comparison_log(results: list[ModelResult], generated_at: str) -> None:
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
        "",
        "## Samlet oversikt",
        "",
        "| Modell | Status | MAE | RMSE | Kommentar |",
        "| --- | --- | --- | --- | --- |",
    ]

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
    for model_name in MODEL_METADATA:
        display_name = MODEL_METADATA[model_name]["display_name"]
        content_lines.append(
            f"- [{display_name} kode]({display_name} kode.md) og "
            f"[{display_name} resultater]({display_name} resultater.md)"
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

    content_lines.extend(
        [
            "- Resultatene må tolkes som foreløpige fordi datasettet per nå bare dekker 9 måneder.",
            "",
            "## Notater Til Rapport",
            "",
            "- Bruk tabellen over direkte som grunnlag for sammenligning av modellprestasjon.",
            "- Beskriv at SARIMA ikke kunne evalueres faglig forsvarlig med dagens tidsdybde.",
            "- Diskuter om lav MAE alene er nok, eller om modellens tolkbarhet og datakrav også bør vektlegges.",
        ]
    )

    COMPARISON_LOG_PATH.write_text("\n".join(content_lines) + "\n", encoding="utf-8")


def save_model_specific_outputs(
    results: list[ModelResult],
    prediction_frames: dict[str, pd.DataFrame],
) -> None:
    generated_at = datetime.now().isoformat(timespec="seconds")
    for model_name in MODEL_METADATA:
        write_model_code_log(model_name)

        metrics_path = RESULTS_DIR / f"{model_name}_metrics.json"
        predictions_path = RESULTS_DIR / f"{model_name}_predictions.csv"

        result = next((item for item in results if item.model == model_name), None)
        pred_df = prediction_frames.get(model_name, pd.DataFrame())

        if result is None:
            write_model_result_log(
                ModelResult(model=model_name, status="not_run", details={}),
                pred_df,
                generated_at,
            )
            metrics_path.unlink(missing_ok=True)
            predictions_path.unlink(missing_ok=True)
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

        write_model_result_log(result, pred_df, generated_at)

    write_model_comparison_log(results, generated_at)


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, sep=";", encoding="utf-8-sig")
    df.columns = [col.strip() for col in df.columns]
    df["Fartøy/Måned"] = df["Fartøy/Måned"].astype(str).str.strip()

    long_df = df.melt(
        id_vars=["Fartøy/Måned", "Spesielle behov/krav"],
        value_vars=MONTH_COLUMNS,
        var_name="month_name",
        value_name="offhire_days_raw",
    )

    long_df = long_df[long_df["offhire_days_raw"].notna()].copy()
    long_df["offhire_days_raw"] = long_df["offhire_days_raw"].astype(str).str.strip()
    long_df = long_df[long_df["offhire_days_raw"].str.lower().ne("nan")].copy()
    long_df = long_df[long_df["offhire_days_raw"].ne("N/A")].copy()
    long_df = long_df[long_df["offhire_days_raw"].ne("")].copy()

    long_df["offhire_days"] = (
        long_df["offhire_days_raw"].str.replace(",", ".", regex=False).astype(float)
    )
    long_df["month_num"] = long_df["month_name"].map(MONTH_ORDER)
    long_df["date"] = pd.to_datetime(
        {
            "year": np.full(len(long_df), SYNTHETIC_YEAR, dtype=int),
            "month": long_df["month_num"].astype(int),
            "day": 1,
        }
    )
    long_df = long_df.rename(columns={"Fartøy/Måned": "vessel"})
    long_df = long_df.sort_values(["vessel", "date"]).reset_index(drop=True)
    long_df["vessel"] = long_df["vessel"].str.replace(r"\s+", " ", regex=True).str.strip()
    return long_df[
        ["vessel", "month_name", "month_num", "date", "offhire_days", "Spesielle behov/krav"]
    ]


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


def train_test_split_panel(features_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_dates = sorted(features_df["date"].unique())
    if len(unique_dates) < 4:
        raise DataTooShortError(
            "Det finnes for få tidssteg etter feature engineering til å lage train/test-splitt."
        )
    test_dates = set(unique_dates[-2:])
    train_df = features_df[~features_df["date"].isin(test_dates)].copy()
    test_df = features_df[features_df["date"].isin(test_dates)].copy()
    if train_df.empty or test_df.empty:
        raise DataTooShortError("Train/test-splitt ble tom.")
    return train_df, test_df


def run_exponential_smoothing(panel_df: pd.DataFrame) -> tuple[ModelResult, pd.DataFrame]:
    predictions: list[dict[str, Any]] = []
    for vessel, vessel_df in panel_df.groupby("vessel"):
        vessel_df = vessel_df.sort_values("date").reset_index(drop=True)
        if len(vessel_df) < 6:
            continue

        train_df = vessel_df.iloc[:-2]
        test_df = vessel_df.iloc[-2:]
        if len(train_df) < 4 or test_df.empty:
            continue

        train_series = train_df["offhire_days"].astype(float)
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
            forecast = np.asarray(fit.forecast(len(test_df)), dtype=float)
            if not np.isfinite(forecast).all():
                raise ValueError("Forecast contains non-finite values.")
        except Exception:
            # Fallback for very short or numerically unstable vessel series.
            forecast = np.repeat(float(train_series.iloc[-1]), len(test_df))
        forecast = np.clip(forecast, 0.0, None)

        for date_value, actual, pred in zip(test_df["date"], test_df["offhire_days"], forecast):
            predictions.append(
                {
                    "model": "exponential_smoothing",
                    "vessel": vessel,
                    "date": date_value.strftime("%Y-%m-%d"),
                    "actual": float(actual),
                    "prediction": float(pred),
                }
            )

    if not predictions:
        return (
            ModelResult(
                model="exponential_smoothing",
                status="skipped",
                details={"reason": "For få observasjoner per fartøy til å kjøre eksponentiell glatting."},
            ),
            pd.DataFrame(),
        )

    pred_df = pd.DataFrame(predictions)
    metrics = ModelResult(
        model="exponential_smoothing",
        status="ok",
        mae=float(mean_absolute_error(pred_df["actual"], pred_df["prediction"])),
        rmse=rmse(pred_df["actual"].to_numpy(), pred_df["prediction"].to_numpy()),
        details={"vessels_used": int(pred_df["vessel"].nunique())},
    )
    return metrics, pred_df


def run_sarima(panel_df: pd.DataFrame) -> tuple[ModelResult, pd.DataFrame]:
    fleet_series = (
        panel_df.groupby("date", as_index=True)["offhire_days"].sum().sort_index()
    )
    if len(fleet_series) < 24:
        return (
            ModelResult(
                model="sarima",
                status="skipped",
                details={
                    "reason": (
                        "SARIMA ble ikke kjørt fordi dataserien er for kort for en forsvarlig "
                        "månedlig sesongmodell. Minst 24 observasjoner anbefales."
                    ),
                    "observations": int(len(fleet_series)),
                },
            ),
            pd.DataFrame(),
        )

    train = fleet_series.iloc[:-2]
    test = fleet_series.iloc[-2:]

    model = SARIMAX(
        train,
        order=(1, 0, 1),
        seasonal_order=(1, 0, 0, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)
    forecast = np.clip(np.asarray(fit.forecast(steps=len(test)), dtype=float), 0.0, None)
    pred_df = pd.DataFrame(
        {
            "model": "sarima",
            "vessel": "fleet_total",
            "date": test.index.strftime("%Y-%m-%d"),
            "actual": test.values.astype(float),
            "prediction": forecast,
        }
    )
    metrics = ModelResult(
        model="sarima",
        status="ok",
        mae=float(mean_absolute_error(pred_df["actual"], pred_df["prediction"])),
        rmse=rmse(pred_df["actual"].to_numpy(), pred_df["prediction"].to_numpy()),
        details={"series_type": "fleet_total"},
    )
    return metrics, pred_df


def run_xgboost(features_df: pd.DataFrame) -> tuple[ModelResult, pd.DataFrame]:
    from xgboost import XGBRegressor

    train_df, test_df = train_test_split_panel(features_df)
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

    feature_columns = numeric_features + categorical_features
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
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
        },
    )
    return metrics, pred_df


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


def save_outputs(
    results: list[ModelResult],
    prediction_frames: dict[str, pd.DataFrame],
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

    save_model_specific_outputs(results, prediction_frames)

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


def main() -> None:
    ensure_results_dir()
    panel_df = load_dataset()

    results: list[ModelResult] = []
    prediction_frames: dict[str, pd.DataFrame] = {}

    features_df = build_panel_features(panel_df, lags=3)

    model_runners = [
        ("sarima", lambda: run_sarima(panel_df)),
        ("exponential_smoothing", lambda: run_exponential_smoothing(panel_df)),
        ("xgboost", lambda: run_xgboost(features_df)),
        ("lstm", lambda: run_lstm(panel_df)),
    ]

    for model_name, runner in model_runners:
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

    save_outputs(results, prediction_frames)

    print("Modellkjøring fullført.")
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
