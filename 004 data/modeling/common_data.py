from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common_config import (
    MASTER_DATA_PATH,
    MAX_FUTURE_FORECAST_HORIZON,
    MONTH_COLUMNS,
    MONTH_NAME_BY_NUMBER,
    MONTH_ORDER,
    TEST_DATA_PATH,
    TEST_START_DATE,
    TRAIN_DATA_PATH,
    TRAIN_END_DATE,
    YEAR_PATTERN,
)
from common_types import DataTooShortError


def format_period(panel_df: pd.DataFrame) -> str:
    if panel_df.empty:
        return "ukjent"
    return f"{panel_df['date'].min():%Y-%m} til {panel_df['date'].max():%Y-%m}"


def format_forecast_horizon(months: int) -> str:
    if months == 1:
        return "1 måned"
    return f"{months} måneder"


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


def future_dates_from_panel(
    panel_df: pd.DataFrame,
    horizon: int = MAX_FUTURE_FORECAST_HORIZON,
) -> list[pd.Timestamp]:
    if panel_df.empty:
        raise DataTooShortError("Datasettet er tomt, så fremtidsdatoer kan ikke bygges.")

    start = panel_df["date"].max() + pd.offsets.MonthBegin(1)
    return list(pd.date_range(start=start, periods=horizon, freq="MS"))
