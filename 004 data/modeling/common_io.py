from __future__ import annotations

import inspect
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from common_config import (
    COMPARISON_LOG_PATH,
    FUTURE_FORECAST_HORIZONS,
    GENERATE_FUTURE_FORECASTS,
    MAX_FUTURE_FORECAST_HORIZON,
    MODEL_ARTIFACTS_DIR,
    MODEL_LOG_DIR,
    MODEL_METADATA,
    MONTH_NAME_BY_NUMBER,
    RESULTS_DIR,
)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common_data import format_forecast_horizon
from common_eval import build_metrics_table
from common_types import ModelResult


def ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_LOG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "figures").mkdir(parents=True, exist_ok=True)


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


def write_model_code_log(
    model_name: str,
    function_name: str,
    function_obj: Callable[..., Any],
) -> None:
    metadata = MODEL_METADATA[model_name]
    function_source = inspect.getsource(function_obj).rstrip()
    file_path = model_artifact_path(model_name, "kode.md")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    module_name = getattr(function_obj, "__module__", "ukjent")
    content = (
        f"# {metadata['display_name']} kode\n\n"
        "Dette dokumentet genereres automatisk fra den aktive implementasjonen i "
        f"`004 data/modeling/{module_name}.py`.\n\n"
        f"- Modell: `{model_name}`\n"
        f"- Funksjon: `{function_name}`\n\n"
        "```python\n"
        f"{function_source}\n"
        "```\n"
    )
    file_path.write_text(content, encoding="utf-8")


def format_metric(value: float | None) -> str:
    if value is None:
        return "ikke tilgjengelig"
    return f"{value:.4f}"


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
    if not future_df.empty:
        content_lines.extend(
            [
                (
                    f"- Fremtidsprognoser: `{int(future_df['forecast_step'].max())}` steg "
                    f"fra `{pd.to_datetime(future_df['date']).min():%Y-%m}` "
                    f"til `{pd.to_datetime(future_df['date']).max():%Y-%m}`"
                ),
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


def summarize_result_details(result: ModelResult) -> str:
    details = result.details or {}
    if result.status != "ok":
        return str(details.get("reason", "")).replace("|", "\\|")

    if result.model == "exponential_smoothing" and "vessels_used" in details:
        return f"Fartøy brukt: {details['vessels_used']}"
    if result.model == "xgboost":
        return f"Walk-forward måneder: {details.get('walk_forward_steps', 'ukjent')}"
    if result.model == "lstm":
        return f"Sekvenslengde: {details.get('sequence_length', 'ukjent')}"
    if result.model == "sarima":
        return f"Modellerte fartøy: {details.get('modeled_vessels', 'ukjent')}"
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
    pipeline: Any,
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


def cleanup_future_output_artifacts() -> None:
    future_output_paths = [
        RESULTS_DIR / "future_predictions.csv",
        RESULTS_DIR / "future_totals_by_model_and_date.csv",
    ]
    for horizon in FUTURE_FORECAST_HORIZONS:
        future_output_paths.extend(
            [
                RESULTS_DIR / f"future_predictions_{horizon}m.csv",
                RESULTS_DIR / f"future_predictions_{horizon}m_pivot.csv",
                RESULTS_DIR / "figures" / f"future_total_offhire_{horizon}m.png",
            ]
        )

    for path in future_output_paths:
        path.unlink(missing_ok=True)
        if path.suffix == ".csv":
            path.with_suffix(".md").unlink(missing_ok=True)


def expand_future_predictions_by_horizon(
    future_df: pd.DataFrame,
    horizons: list[int],
) -> pd.DataFrame:
    if future_df.empty:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for horizon in horizons:
        subset = future_df[future_df["forecast_step"] <= horizon].copy()
        if subset.empty:
            continue
        subset.insert(2, "forecast_horizon", horizon)
        frames.append(subset)

    if not frames:
        return pd.DataFrame()

    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["forecast_horizon", "model", "vessel", "forecast_step"])
        .reset_index(drop=True)
    )


def save_future_forecast_artifacts(
    future_prediction_frames: dict[str, pd.DataFrame],
) -> None:
    non_empty_future_frames = [
        frame for frame in future_prediction_frames.values() if not frame.empty
    ]
    if not non_empty_future_frames:
        cleanup_future_output_artifacts()
        return

    combined_future_base = pd.concat(non_empty_future_frames, ignore_index=True)
    combined_future_base = combined_future_base.sort_values(
        ["model", "vessel", "forecast_step"]
    ).reset_index(drop=True)
    combined_future = expand_future_predictions_by_horizon(
        combined_future_base,
        FUTURE_FORECAST_HORIZONS,
    )

    if combined_future.empty:
        cleanup_future_output_artifacts()
        return

    display_map = {key: value["display_name"] for key, value in MODEL_METADATA.items()}
    write_dataframe_artifacts(
        combined_future,
        RESULTS_DIR / "future_predictions.csv",
        "Fremtidsprognoser for alle modeller og horisonter",
    )

    totals_df = (
        combined_future.assign(display_name=lambda df: df["model"].map(display_map))
        .groupby(["forecast_horizon", "date", "display_name"], as_index=False)
        .agg(total_prediction=("prediction", "sum"))
        .sort_values(["forecast_horizon", "date", "display_name"])
        .reset_index(drop=True)
    )
    write_dataframe_artifacts(
        totals_df,
        RESULTS_DIR / "future_totals_by_model_and_date.csv",
        "Samlet prognostisert offhire per dato og modell",
    )

    for horizon in FUTURE_FORECAST_HORIZONS:
        horizon_df = combined_future[combined_future["forecast_horizon"] == horizon].copy()
        if horizon_df.empty:
            continue

        write_dataframe_artifacts(
            horizon_df,
            RESULTS_DIR / f"future_predictions_{horizon}m.csv",
            f"Fremtidsprognoser {format_forecast_horizon(horizon)} fram",
        )

        pivot_df = (
            horizon_df.assign(display_name=lambda df: df["model"].map(display_map))
            .pivot(index=["vessel", "date"], columns="display_name", values="prediction")
            .reset_index()
            .sort_values(["vessel", "date"])
            .reset_index(drop=True)
        )
        pivot_df.columns.name = None
        write_dataframe_artifacts(
            pivot_df,
            RESULTS_DIR / f"future_predictions_{horizon}m_pivot.csv",
            f"Fremtidsprognoser {format_forecast_horizon(horizon)} fram per fartøy",
        )

        totals_horizon = totals_df[totals_df["forecast_horizon"] == horizon].copy()
        totals_horizon["date"] = pd.to_datetime(totals_horizon["date"])
        fig, ax = plt.subplots(figsize=(9, 4.8))
        if totals_horizon["date"].nunique() == 1:
            ordered = totals_horizon.sort_values("total_prediction", ascending=False)
            ax.bar(
                ordered["display_name"],
                ordered["total_prediction"],
                color="#2C7A7B",
            )
            ax.set_xlabel("Modell")
        else:
            for display_name, model_df in totals_horizon.groupby("display_name"):
                ax.plot(
                    model_df["date"],
                    model_df["total_prediction"],
                    marker="o",
                    linewidth=1.8,
                    label=display_name,
                )
            ax.legend()
            ax.set_xlabel("Dato")
        ax.set_title(f"Samlet prognostisert offhire {format_forecast_horizon(horizon)} fram")
        ax.set_ylabel("Prognostisert offhire (%)")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "figures" / f"future_total_offhire_{horizon}m.png", dpi=200)
        plt.close(fig)


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
    if GENERATE_FUTURE_FORECASTS and not panel_df.empty:
        future_start = panel_df["date"].max() + pd.offsets.MonthBegin(1)
        future_end = future_start + pd.offsets.MonthBegin(MAX_FUTURE_FORECAST_HORIZON - 1)
        horizons_text = ", ".join(
            format_forecast_horizon(horizon) for horizon in FUTURE_FORECAST_HORIZONS
        )
        content_lines.extend(
            [
                f"- Fremtidsprognoser: `{horizons_text}`",
                f"- Prognosevindu: `{future_start:%Y-%m}` til `{future_end:%Y-%m}`",
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
    model_functions: dict[str, Callable[..., Any]],
) -> None:
    generated_at = datetime.now().isoformat(timespec="seconds")
    for model_name in active_models:
        write_model_code_log(
            model_name,
            MODEL_METADATA[model_name]["function_name"],
            model_functions[model_name],
        )

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
