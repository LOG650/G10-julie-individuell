from __future__ import annotations

import inspect
import json
import os
import re
import tempfile
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "g10-julie-individuell-matplotlib"),
)

import matplotlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    module=r"statsmodels\.tsa\.holtwinters\.model",
)
warnings.filterwarnings(
    "ignore",
    category=ConvergenceWarning,
    module=r"statsmodels\..*",
)
warnings.filterwarnings(
    "ignore",
    category=ValueWarning,
    module=r"statsmodels\..*",
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
MAX_OFFHIRE_VALUE = 100.0

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
ACTIVE_MODELS = ["sarima", "exponential_smoothing", "xgboost", "lstm"]
GENERATE_FUTURE_FORECASTS = False


@dataclass
class ModelResult:
    model: str
    status: str
    mae: float | None = None
    rmse: float | None = None
    smape: float | None = None
    details: dict[str, Any] | None = None


class DataTooShortError(RuntimeError):
    pass


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


def ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_LOG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "figures").mkdir(parents=True, exist_ok=True)


def format_period(panel_df: pd.DataFrame) -> str:
    if panel_df.empty:
        return "ukjent"
    return f"{panel_df['date'].min():%Y-%m} til {panel_df['date'].max():%Y-%m}"


def build_split_metadata(
    train_panel: pd.DataFrame,
    test_panel: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "evaluation_train_period": format_period(train_panel),
        "evaluation_test_period": format_period(test_panel),
        "train_observations": int(len(train_panel)),
        "test_observations": int(len(test_panel)),
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


def normalize_optional_string(value: Any) -> str | None:
    if pd.isna(value):
        return None
    return str(value)


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


def write_dataframe_artifacts(df: pd.DataFrame, csv_path: Path, markdown_title: str) -> None:
    if df.empty:
        csv_path.unlink(missing_ok=True)
        csv_path.with_suffix(".md").unlink(missing_ok=True)
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    markdown_content = [
        f"# {markdown_title}",
        "",
        dataframe_to_markdown(df, max_rows=len(df)),
        "",
    ]
    csv_path.with_suffix(".md").write_text("\n".join(markdown_content), encoding="utf-8")


def model_method_path(model_name: str) -> Path:
    return model_artifact_path(model_name, "metode.md")


def model_extra_artifact_paths(model_name: str) -> list[Path]:
    paths = [model_method_path(model_name)]
    if model_name == "sarima":
        paths.extend(
            [
                model_artifact_path(model_name, "stasjonaritet.csv"),
                model_artifact_path(model_name, "stasjonaritet.md"),
                model_artifact_path(model_name, "kandidatmodeller.csv"),
                model_artifact_path(model_name, "kandidatmodeller.md"),
                model_artifact_path(model_name, "modellvalg_per_fartoy.csv"),
                model_artifact_path(model_name, "modellvalg_per_fartoy.md"),
                model_artifact_path(model_name, "residualdiagnostikk.csv"),
                model_artifact_path(model_name, "residualdiagnostikk.md"),
                model_artifact_path(model_name, "acf.png"),
                model_artifact_path(model_name, "pacf.png"),
                model_artifact_path(model_name, "residualdiagnostikk.png"),
                model_artifact_path(model_name, "representativ_testplot.png"),
            ]
        )
    elif model_name == "exponential_smoothing":
        paths.extend(
            [
                model_artifact_path(model_name, "modellvalg_per_fartoy.csv"),
                model_artifact_path(model_name, "modellvalg_per_fartoy.md"),
                model_artifact_path(model_name, "residualdiagnostikk.csv"),
                model_artifact_path(model_name, "residualdiagnostikk.md"),
                model_artifact_path(model_name, "representativ_testplot.png"),
            ]
        )
    elif model_name == "xgboost":
        paths.extend(
            [
                model_artifact_path(model_name, "feature_importance.csv"),
                model_artifact_path(model_name, "feature_importance.md"),
                model_artifact_path(model_name, "feature_importance.png"),
                model_artifact_path(model_name, "representativ_testplot.png"),
            ]
        )
    elif model_name == "lstm":
        paths.extend(
            [
                model_artifact_path(model_name, "training_history.csv"),
                model_artifact_path(model_name, "training_history.md"),
                model_artifact_path(model_name, "training_history.png"),
                model_artifact_path(model_name, "representativ_testplot.png"),
            ]
        )
    return paths


def cleanup_extra_artifacts(model_name: str) -> None:
    for artifact_path in model_extra_artifact_paths(model_name):
        artifact_path.unlink(missing_ok=True)


def format_order_tuple(values: tuple[int, ...]) -> str:
    return "(" + ", ".join(str(value) for value in values) + ")"


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


def write_model_method_log(result: ModelResult) -> None:
    metadata = MODEL_METADATA[result.model]
    details = result.details or {}
    file_path = model_method_path(result.model)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    header_lines = [
        f"# {metadata['display_name']} metode",
        "",
        f"- Modell: `{result.model}`",
        f"- Evalueringsnivå: `{details.get('evaluation_level', 'ikke oppgitt')}`",
        f"- Train: `{details.get('evaluation_train_period', 'ikke oppgitt')}`",
        f"- Test: `{details.get('evaluation_test_period', 'ikke oppgitt')}`",
        "",
    ]

    step_lines: list[str]
    if result.status != "ok":
        step_lines = [
            "## Status",
            "",
            f"Modellen ble ikke fullført i siste kjøring. Årsak: `{details.get('reason', 'ukjent')}`.",
        ]
        file_path.write_text("\n".join(header_lines + step_lines) + "\n", encoding="utf-8")
        return

    if result.model == "sarima":
        selected_order = details.get("selected_order", {})
        seasonal_order = details.get("selected_seasonal_order", {})
        step_lines = [
            "## Steg 1. Datagrunnlag",
            "",
            (
                "Månedlige tidsserier ble bygget per fartøy fra train-settet. "
                f"Train-seriene dekker `{details.get('evaluation_train_period', 'ukjent')}`."
            ),
            "",
            "## Steg 2. Stasjonaritet",
            "",
            (
                "ADF-test ble brukt som støtte for differensieringsvalg per fartøy. "
                "Endelig modellvalg ble deretter gjort gjennom et begrenset ARIMA/SARIMA-søk."
            ),
            "",
            "## Steg 3. ACF- og PACF-analyse",
            "",
            (
                "ACF og PACF ble lagret for et representativt fartøy som diagnostisk støtte. "
                "Se `acf.png`, `pacf.png` og `kandidatmodeller.md`."
            ),
            "",
            "## Steg 4. Modellestimering",
            "",
            (
                "Kandidatmodeller ble estimert per fartøy og rangert med AIC, BIC og parsimoni. "
                "Valgte spesifikasjoner er oppsummert i `modellvalg_per_fartoy.md`."
            ),
            "",
            "## Steg 5. Modellvalidering",
            "",
            (
                "Residualdiagnostikk ble gjennomført per fartøy med Ljung-Box-test, "
                "og modellene ble evaluert med ekspanderende 1-stegs prognoser på testperioden. "
                f"Samlet resultat: `MAE={format_metric(result.mae)}`, "
                f"`RMSE={format_metric(result.rmse)}` og `sMAPE={format_metric(result.smape)}`."
            ),
            "",
            "## Repo-artefakter",
            "",
            "- `stasjonaritet.md`",
            "- `kandidatmodeller.md`",
            "- `modellvalg_per_fartoy.md`",
            "- `residualdiagnostikk.md`",
            "- `acf.png`",
            "- `pacf.png`",
            "- `residualdiagnostikk.png`",
            "- `representativ_testplot.png`",
        ]
    elif result.model == "exponential_smoothing":
        step_lines = [
            "## Steg 1. Datagrunnlag",
            "",
            "Egne månedlige tidsserier per fartøy bygget fra train-settet.",
            "",
            "## Steg 2. Mønsteranalyse",
            "",
            (
                "Seriene ble vurdert som korte, nulltunge og ujevne. Det ble derfor ikke brukt "
                "egen sesongkomponent i modellen."
            ),
            "",
            "## Steg 3. Modellspesifikasjon",
            "",
            (
                "Det ble sammenlignet et begrenset sett av additive ETS-varianter per fartøy. "
                "Valgt spesifikasjon per fartøy er lagret i `modellvalg_per_fartoy.md`."
            ),
            "",
            "## Steg 4. Modellestimering",
            "",
            "Hver fartøysserie ble estimert separat på train-settet, og beste ETS-variant ble beholdt.",
            "",
            "## Steg 5. Modellvalidering",
            "",
            (
                "Modellen ble evaluert med ekspanderende 1-stegs prognoser gjennom testperioden. "
                f"Resultat: `MAE={format_metric(result.mae)}`, `RMSE={format_metric(result.rmse)}` "
                f"og `sMAPE={format_metric(result.smape)}`."
            ),
            "",
            "## Repo-artefakter",
            "",
            "- `modellvalg_per_fartoy.md`",
            "- `residualdiagnostikk.md`",
            "- `representativ_testplot.png`",
        ]
    elif result.model == "xgboost":
        step_lines = [
            "## Steg 1. Datagrunnlag",
            "",
            "Paneldata med én rad per fartøy og måned, bygget fra train/test-splittet.",
            "",
            "## Steg 2. Featureanalyse",
            "",
            (
                "Modellen bruker laggede offhire-verdier, rullerende statistikk, kalendervariabler, "
                "fartøy-ID og flagg for spesielle behov."
            ),
            "",
            "## Steg 3. Modellspesifikasjon",
            "",
            (
                "XGBoost ble satt opp som en gradient boosted tree-modell med et fast feature-set og "
                "forhåndsvalgte hyperparametre for dybde, læringsrate og antall trær."
            ),
            "",
            "## Steg 4. Modellestimering",
            "",
            "Modellen ble re-trent måned for måned i en ekspanderende 1-stegs evaluering.",
            "",
            "## Steg 5. Modellvalidering",
            "",
            (
                f"Modellen ble evaluert på testperioden med `MAE={format_metric(result.mae)}`, "
                f"`RMSE={format_metric(result.rmse)}` og `sMAPE={format_metric(result.smape)}`."
            ),
            "",
            "## Repo-artefakter",
            "",
            "- `feature_importance.md`",
            "- `feature_importance.png`",
            "- `representativ_testplot.png`",
        ]
    else:
        step_lines = [
            "## Steg 1. Datagrunnlag",
            "",
            "Sekvensdata bygget fra historiske observasjoner per fartøy.",
            "",
            "## Steg 2. Sekvensanalyse",
            "",
            (
                "Historikken ble omsatt til sekvenser med tolv tidligere tidssteg, der både nivå, "
                "månedssyklus og spesialkrav inngår som input."
            ),
            "",
            "## Steg 3. Modellspesifikasjon",
            "",
            (
                "LSTM-modellen ble definert med ett LSTM-lag, ett tett lag og standardisering "
                "av både input og målvariabel basert på treningsdata."
            ),
            "",
            "## Steg 4. Modellestimering",
            "",
            "Modellen ble re-trent måned for måned i en ekspanderende 1-stegs evaluering.",
            "",
            "## Steg 5. Modellvalidering",
            "",
            (
                f"Holdout-evaluering på testsekvenser ga `MAE={format_metric(result.mae)}`, "
                f"`RMSE={format_metric(result.rmse)}` og `sMAPE={format_metric(result.smape)}`."
            ),
            "",
            "## Repo-artefakter",
            "",
            "- `training_history.md`",
            "- `training_history.png`",
            "- `representativ_testplot.png`",
        ]

    file_path.write_text("\n".join(header_lines + step_lines) + "\n", encoding="utf-8")


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
        (
            f"- sMAPE: `{result.smape:.4f}`"
            if result.smape is not None
            else "- sMAPE: `ikke tilgjengelig`"
        ),
    ]
    if "evaluation_train_period" in details and "evaluation_test_period" in details:
        content_lines.extend(
            [
                f"- Evaluering train: `{details['evaluation_train_period']}`",
                f"- Evaluering test: `{details['evaluation_test_period']}`",
            ]
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
            "## Testprediksjoner",
            "",
            predictions_preview,
        ]
    )
    if not pred_df.empty and len(pred_df) > 10:
        content_lines.extend(
            [
                "",
                f"Viser de første 10 av totalt {len(pred_df)} testprediksjoner.",
            ]
        )
    if not future_df.empty:
        content_lines.extend(
            [
                "",
                "## Fremtidsprognoser",
                "",
                future_preview,
            ]
        )
        if len(future_df) > 10:
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
            f"Walk-forward måneder: {details.get('walk_forward_steps', 'ukjent')}"
        )
    if result.model == "lstm":
        return (
            f"Sekvenslengde: {details.get('sequence_length', 'ukjent')}"
        )
    if result.model == "sarima":
        return (
            f"Modellerte fartøy: {details.get('modeled_vessels', 'ukjent')}"
        )
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


def save_representative_prediction_plot(
    model_name: str,
    representative_vessel: str | None,
    train_panel: pd.DataFrame,
    test_panel: pd.DataFrame,
    pred_df: pd.DataFrame,
    title: str,
) -> None:
    if representative_vessel is None or pred_df.empty:
        return

    train_series = (
        train_panel[train_panel["vessel"] == representative_vessel]
        .sort_values("date")
        .copy()
    )
    test_series = (
        test_panel[test_panel["vessel"] == representative_vessel]
        .sort_values("date")
        .copy()
    )
    pred_series = (
        pred_df[pred_df["vessel"] == representative_vessel]
        .sort_values("date")
        .copy()
    )
    if train_series.empty or test_series.empty or pred_series.empty:
        return

    pred_series["date"] = pd.to_datetime(pred_series["date"])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        train_series["date"],
        train_series["offhire_days"],
        color="#6B7280",
        linewidth=1.8,
        label="Train faktisk",
    )
    ax.plot(
        test_series["date"],
        test_series["offhire_days"],
        color="#0F4C5C",
        linewidth=2.0,
        marker="o",
        label="Test faktisk",
    )
    ax.plot(
        pred_series["date"],
        pred_series["prediction"],
        color="#D97706",
        linewidth=2.0,
        marker="o",
        label="Predikert",
    )
    ax.set_title(title)
    ax.set_xlabel("Dato")
    ax.set_ylabel("Offhire (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(model_artifact_path(model_name, "representativ_testplot.png"), dpi=200)
    plt.close(fig)


def save_feature_importance_artifacts(
    pipeline: Pipeline,
    top_n: int = 15,
) -> None:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    importance_df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    write_dataframe_artifacts(
        importance_df,
        model_artifact_path("xgboost", "feature_importance.csv"),
        "XGBoost feature importance",
    )

    top_df = importance_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(top_df["feature"], top_df["importance"], color="#2C7A7B")
    ax.set_title("XGBoost feature importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(model_artifact_path("xgboost", "feature_importance.png"), dpi=200)
    plt.close(fig)


def save_lstm_training_history(history_df: pd.DataFrame) -> None:
    if history_df.empty:
        return
    write_dataframe_artifacts(
        history_df,
        model_artifact_path("lstm", "training_history.csv"),
        "LSTM training history",
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(history_df["epoch"], history_df["loss"], label="Train loss", color="#0F4C5C")
    if "val_loss" in history_df.columns and history_df["val_loss"].notna().any():
        ax.plot(
            history_df["epoch"],
            history_df["val_loss"],
            label="Val loss",
            color="#D97706",
        )
    ax.set_title("LSTM training history")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(model_artifact_path("lstm", "training_history.png"), dpi=200)
    plt.close(fig)


def save_summary_artifacts(prediction_frames: dict[str, pd.DataFrame]) -> None:
    non_empty_frames = [frame for frame in prediction_frames.values() if not frame.empty]
    if not non_empty_frames:
        return

    combined = pd.concat(non_empty_frames, ignore_index=True)
    overall = build_metrics_table(combined, ["model"])
    by_vessel = build_metrics_table(combined, ["model", "vessel"])
    by_month = build_metrics_table(combined, ["model", "date"])

    write_dataframe_artifacts(
        overall,
        RESULTS_DIR / "model_comparison_summary.csv",
        "Samlet modelltest",
    )
    write_dataframe_artifacts(
        by_vessel,
        RESULTS_DIR / "metrics_by_vessel.csv",
        "Testmetrikker per fartøy",
    )
    write_dataframe_artifacts(
        by_month,
        RESULTS_DIR / "metrics_by_month.csv",
        "Testmetrikker per måned",
    )

    display_map = {key: value["display_name"] for key, value in MODEL_METADATA.items()}
    overall["display_name"] = overall["model"].map(display_map)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ordered = overall.sort_values("mae")
    ax.bar(ordered["display_name"], ordered["mae"], color="#2C7A7B")
    ax.set_title("MAE per modell i testperioden")
    ax.set_ylabel("MAE")
    ax.set_xlabel("Modell")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figures" / "mae_per_model.png", dpi=200)
    plt.close(fig)

    if not by_month.empty:
        by_month["date"] = pd.to_datetime(by_month["date"])
        fig, ax = plt.subplots(figsize=(10, 5))
        for model_name, month_df in by_month.groupby("model"):
            ax.plot(
                month_df["date"],
                month_df["mae"],
                marker="o",
                linewidth=1.8,
                label=display_map.get(model_name, model_name),
            )
        ax.set_title("MAE per testmåned")
        ax.set_xlabel("Dato")
        ax.set_ylabel("MAE")
        ax.legend()
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "figures" / "mae_by_month.png", dpi=200)
        plt.close(fig)

    if not by_vessel.empty:
        heatmap_df = (
            by_vessel.copy()
            .assign(display_name=lambda df: df["model"].map(display_map))
            .pivot(index="vessel", columns="display_name", values="mae")
            .sort_index()
        )
        fig, ax = plt.subplots(figsize=(7.5, 7))
        image = ax.imshow(heatmap_df.to_numpy(), aspect="auto", cmap="YlOrRd")
        ax.set_title("MAE per fartøy og modell")
        ax.set_xticks(np.arange(len(heatmap_df.columns)))
        ax.set_xticklabels(heatmap_df.columns, rotation=30, ha="right")
        ax.set_yticks(np.arange(len(heatmap_df.index)))
        ax.set_yticklabels(heatmap_df.index)
        fig.colorbar(image, ax=ax, label="MAE")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "figures" / "mae_heatmap_by_vessel.png", dpi=200)
        plt.close(fig)


def future_dates_from_panel(panel_df: pd.DataFrame, horizon: int = 2) -> list[pd.Timestamp]:
    if panel_df.empty:
        raise DataTooShortError("Datasettet er tomt, så fremtidsdatoer kan ikke bygges.")

    start = panel_df["date"].max() + pd.offsets.MonthBegin(1)
    return list(pd.date_range(start=start, periods=horizon, freq="MS"))


def write_model_comparison_log(
    results: list[ModelResult],
    generated_at: str,
    panel_df: pd.DataFrame,
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
            ]
        )
    content_lines.extend(
        [
            "",
            "## Samlet oversikt",
            "",
            "| Modell | Status | MAE | RMSE | sMAPE | Kommentar |",
            "| --- | --- | --- | --- | --- | --- |",
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
                    format_metric(result.smape),
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
            f"- [{display_name} metode](../models/{display_name}/metode.md), "
            f"[{display_name} kode](../models/{display_name}/kode.md) og "
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

        write_model_method_log(result)
        metrics_payload = {
            "model": result.model,
            "status": result.status,
            "mae": result.mae,
            "rmse": result.rmse,
            "smape": result.smape,
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


def build_vessel_series(panel_df: pd.DataFrame, vessel: str) -> pd.Series:
    vessel_series = (
        panel_df[panel_df["vessel"] == vessel]
        .sort_values("date")
        .set_index("date")["offhire_days"]
        .astype(float)
    )
    vessel_series.index = pd.DatetimeIndex(vessel_series.index)
    return vessel_series.asfreq("MS")


def build_panel_features(
    panel_df: pd.DataFrame,
    lags: tuple[int, ...] = (1, 2, 3, 6, 12),
    rolling_windows: tuple[int, ...] = (3, 6, 12),
) -> pd.DataFrame:
    df = panel_df.copy()
    df["special_flag"] = df["Spesielle behov/krav"].fillna("").str.strip().ne("")
    df["quarter_num"] = ((df["month_num"] - 1) // 3) + 1
    df["year_num"] = df["date"].dt.year
    df["time_idx"] = (
        (df["date"].dt.year - df["date"].dt.year.min()) * 12
        + df["date"].dt.month
        - df["date"].dt.month.min()
    )

    for lag in lags:
        df[f"lag_{lag}"] = df.groupby("vessel")["offhire_days"].shift(lag)

    for window in rolling_windows:
        df[f"rolling_mean_{window}"] = df.groupby("vessel")["offhire_days"].transform(
            lambda series, selected_window=window: series.shift(1)
            .rolling(window=selected_window, min_periods=1)
            .mean()
        )
        df[f"rolling_std_{window}"] = (
            df.groupby("vessel")["offhire_days"].transform(
                lambda series, selected_window=window: series.shift(1)
                .rolling(window=selected_window, min_periods=1)
                .std()
            )
        ).fillna(0.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12.0)

    df = df.dropna(subset=[f"lag_{max(lags)}"]).reset_index(drop=True)
    return df


def split_features_for_evaluation(
    features_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = features_df[features_df["date"] <= TRAIN_END_DATE].copy()
    test_df = features_df[features_df["date"] >= TEST_START_DATE].copy()
    if train_df.empty or test_df.empty:
        raise DataTooShortError("Train/test-splitt ble tom.")
    return train_df, test_df


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


def forecast_exponential_smoothing(panel_df: pd.DataFrame, horizon: int = 2) -> pd.DataFrame:
    future_dates = future_dates_from_panel(panel_df, horizon)
    predictions: list[dict[str, Any]] = []

    for vessel, vessel_df in panel_df.groupby("vessel"):
        vessel_df = vessel_df.sort_values("date").reset_index(drop=True)
        if len(vessel_df) < 12:
            continue

        if vessel_df["offhire_days"].nunique() <= 1:
            forecast = np.repeat(float(vessel_df["offhire_days"].iloc[-1]), horizon)
            for date_value, pred in zip(future_dates, forecast):
                predictions.append(
                    {
                        "model": "exponential_smoothing",
                        "vessel": vessel,
                        "date": date_value.strftime("%Y-%m-%d"),
                        "prediction": float(pred),
                    }
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
    horizon: int = 2,
) -> pd.DataFrame:
    future_dates = future_dates_from_panel(panel_df, horizon)
    predictions: list[dict[str, Any]] = []
    for vessel in sorted(panel_df["vessel"].unique()):
        vessel_series = build_vessel_series(panel_df, vessel).dropna()
        if len(vessel_series) < 24:
            continue

        if vessel_series.nunique() <= 1:
            forecast = np.repeat(float(vessel_series.iloc[-1]), horizon)
            for date_value, pred in zip(future_dates, forecast):
                predictions.append(
                    {
                        "model": "sarima",
                        "vessel": vessel,
                        "date": date_value.strftime("%Y-%m-%d"),
                        "prediction": float(pred),
                    }
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
        for date_value, pred in zip(future_dates, forecast):
            predictions.append(
                {
                    "model": "sarima",
                    "vessel": vessel,
                    "date": date_value.strftime("%Y-%m-%d"),
                    "prediction": float(pred),
                }
            )

    if not predictions:
        return pd.DataFrame()

    return pd.DataFrame(predictions).sort_values(["vessel", "date"]).reset_index(drop=True)


def build_xgboost_pipeline() -> tuple[Pipeline, list[str]]:
    from xgboost import XGBRegressor

    numeric_features = [
        "month_num",
        "quarter_num",
        "year_num",
        "time_idx",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_6",
        "lag_12",
        "rolling_mean_3",
        "rolling_mean_6",
        "rolling_mean_12",
        "rolling_std_3",
        "rolling_std_6",
        "rolling_std_12",
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


def build_xgboost_feature_rows(
    history_panel: pd.DataFrame,
    target_month_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    history_panel = history_panel.sort_values(["vessel", "date"]).reset_index(drop=True)
    start_date = history_panel["date"].min()

    for _, row in target_month_df.iterrows():
        vessel = row["vessel"]
        vessel_history = (
            history_panel[history_panel["vessel"] == vessel]
            .sort_values("date")
            .reset_index(drop=True)
        )
        if len(vessel_history) < 12:
            continue

        history_values = vessel_history["offhire_days"].astype(float).tolist()
        target_date = pd.Timestamp(row["date"])
        special_flag = bool(
            vessel_history["Spesielle behov/krav"].fillna("").astype(str).str.strip().iloc[-1]
        )
        recent_3 = np.asarray(history_values[-3:], dtype=float)
        recent_6 = np.asarray(history_values[-6:], dtype=float)
        recent_12 = np.asarray(history_values[-12:], dtype=float)
        rows.append(
            {
                "month_num": target_date.month,
                "quarter_num": ((target_date.month - 1) // 3) + 1,
                "year_num": target_date.year,
                "time_idx": (target_date.year - start_date.year) * 12 + target_date.month - start_date.month,
                "lag_1": float(history_values[-1]),
                "lag_2": float(history_values[-2]),
                "lag_3": float(history_values[-3]),
                "lag_6": float(history_values[-6]),
                "lag_12": float(history_values[-12]),
                "rolling_mean_3": float(np.mean(recent_3)),
                "rolling_mean_6": float(np.mean(recent_6)),
                "rolling_mean_12": float(np.mean(recent_12)),
                "rolling_std_3": float(np.std(recent_3, ddof=1)) if len(recent_3) > 1 else 0.0,
                "rolling_std_6": float(np.std(recent_6, ddof=1)) if len(recent_6) > 1 else 0.0,
                "rolling_std_12": float(np.std(recent_12, ddof=1)) if len(recent_12) > 1 else 0.0,
                "month_sin": float(np.sin(2 * np.pi * target_date.month / 12.0)),
                "month_cos": float(np.cos(2 * np.pi * target_date.month / 12.0)),
                "vessel": vessel,
                "special_flag": special_flag,
                "date": target_date,
                "actual": float(row["offhire_days"]),
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def run_xgboost(
    panel_df: pd.DataFrame,
    split_metadata: dict[str, Any],
) -> tuple[ModelResult, pd.DataFrame]:
    history_panel = panel_df[panel_df["date"] <= TRAIN_END_DATE].copy()
    test_panel = panel_df[panel_df["date"] >= TEST_START_DATE].copy()
    reference_train_df = build_panel_features(history_panel)
    if reference_train_df.empty:
        raise DataTooShortError("For få historiske observasjoner til å bygge XGBoost-features.")

    reference_pipeline, feature_columns = build_xgboost_pipeline()
    reference_pipeline.fit(reference_train_df[feature_columns], reference_train_df["offhire_days"])
    save_feature_importance_artifacts(reference_pipeline)

    representative_vessel = select_representative_vessel(history_panel)
    predictions: list[dict[str, Any]] = []
    test_dates = sorted(test_panel["date"].unique())

    for date_value in test_dates:
        train_df = build_panel_features(history_panel)
        if train_df.empty:
            continue
        target_month_df = test_panel[test_panel["date"] == date_value].copy()
        feature_rows = build_xgboost_feature_rows(history_panel, target_month_df)
        if feature_rows.empty:
            history_panel = pd.concat([history_panel, target_month_df], ignore_index=True)
            history_panel = history_panel.sort_values(["vessel", "date"]).reset_index(drop=True)
            continue

        pipeline, feature_columns = build_xgboost_pipeline()
        pipeline.fit(train_df[feature_columns], train_df["offhire_days"])
        step_predictions = np.clip(
            pipeline.predict(feature_rows[feature_columns]),
            0.0,
            MAX_OFFHIRE_VALUE,
        )
        for _, row, prediction in zip(feature_rows.index, feature_rows.itertuples(index=False), step_predictions):
            predictions.append(
                {
                    "model": "xgboost",
                    "vessel": row.vessel,
                    "date": pd.Timestamp(row.date).strftime("%Y-%m-%d"),
                    "actual": float(row.actual),
                    "prediction": float(prediction),
                }
            )

        history_panel = pd.concat([history_panel, target_month_df], ignore_index=True)
        history_panel = history_panel.sort_values(["vessel", "date"]).reset_index(drop=True)

    pred_df = pd.DataFrame(predictions)
    save_representative_prediction_plot(
        "xgboost",
        representative_vessel,
        panel_df[panel_df["date"] <= TRAIN_END_DATE],
        test_panel,
        pred_df,
        "XGBoost: representativt testforløp",
    )
    mae_value, rmse_value, smape_value = summarize_prediction_frame(pred_df)
    metrics = ModelResult(
        model="xgboost",
        status="ok",
        mae=mae_value,
        rmse=rmse_value,
        smape=smape_value,
        details={
            **split_metadata,
            "train_rows": int(len(reference_train_df)),
            "test_rows": int(len(pred_df)),
            "walk_forward_steps": int(len(test_dates)),
            "evaluation_method": "ekspanderende 1-stegs prognose med månedlig re-trening",
            "evaluation_level": "fartøynivå",
            "feature_columns": feature_columns,
            "representative_vessel": representative_vessel,
            "model_hyperparameters": {
                "n_estimators": 200,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
            },
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
        if len(vessel_df) < 12:
            continue

        history_values = vessel_df["offhire_days"].astype(float).tolist()
        special_flag = bool(
            vessel_df["Spesielle behov/krav"].fillna("").astype(str).str.strip().iloc[-1]
        )
        start_date = pd.Timestamp(panel_df["date"].min())

        for date_value in future_dates:
            recent_3 = np.asarray(history_values[-3:], dtype=float)
            recent_6 = np.asarray(history_values[-6:], dtype=float)
            recent_12 = np.asarray(history_values[-12:], dtype=float)
            feature_row = pd.DataFrame(
                [
                    {
                        "month_num": date_value.month,
                        "quarter_num": ((date_value.month - 1) // 3) + 1,
                        "year_num": date_value.year,
                        "time_idx": (date_value.year - start_date.year) * 12 + date_value.month - start_date.month,
                        "lag_1": float(history_values[-1]),
                        "lag_2": float(history_values[-2]),
                        "lag_3": float(history_values[-3]),
                        "lag_6": float(history_values[-6]),
                        "lag_12": float(history_values[-12]),
                        "rolling_mean_3": float(np.mean(recent_3)),
                        "rolling_mean_6": float(np.mean(recent_6)),
                        "rolling_mean_12": float(np.mean(recent_12)),
                        "rolling_std_3": float(np.std(recent_3, ddof=1)) if len(recent_3) > 1 else 0.0,
                        "rolling_std_6": float(np.std(recent_6, ddof=1)) if len(recent_6) > 1 else 0.0,
                        "rolling_std_12": float(np.std(recent_12, ddof=1)) if len(recent_12) > 1 else 0.0,
                        "month_sin": float(np.sin(2 * np.pi * date_value.month / 12.0)),
                        "month_cos": float(np.cos(2 * np.pi * date_value.month / 12.0)),
                        "vessel": vessel,
                        "special_flag": special_flag,
                    }
                ]
            )
            prediction = float(
                np.clip(
                    pipeline.predict(feature_row[feature_columns])[0],
                    0.0,
                    MAX_OFFHIRE_VALUE,
                )
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
    panel_df: pd.DataFrame, window_size: int = 12
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


def forecast_lstm(panel_df: pd.DataFrame, horizon: int = 2) -> pd.DataFrame:
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

        for date_value in future_dates:
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
    split_metadata = build_split_metadata(train_panel, test_panel)
    active_models = ACTIVE_MODELS.copy()
    for model_name in active_models:
        cleanup_extra_artifacts(model_name)

    results: list[ModelResult] = []
    prediction_frames: dict[str, pd.DataFrame] = {}
    future_prediction_frames: dict[str, pd.DataFrame] = {
        model_name: pd.DataFrame() for model_name in active_models
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
