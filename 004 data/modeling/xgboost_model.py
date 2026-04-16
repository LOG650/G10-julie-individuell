from __future__ import annotations

from typing import Any

from common_config import (
    MAX_FUTURE_FORECAST_HORIZON,
    MAX_OFFHIRE_VALUE,
    TEST_START_DATE,
    TRAIN_END_DATE,
)
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from common_data import build_panel_features, future_dates_from_panel
from common_eval import (
    build_future_prediction_row,
    select_representative_vessel,
    summarize_prediction_frame,
)
from common_io import save_feature_importance_artifacts, save_representative_prediction_plot
from common_types import DataTooShortError, ModelResult


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
    horizon: int = MAX_FUTURE_FORECAST_HORIZON,
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

        for forecast_step, date_value in enumerate(future_dates, start=1):
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
                build_future_prediction_row(
                    "xgboost",
                    vessel,
                    date_value,
                    prediction,
                    forecast_step,
                )
            )
            history_values.append(prediction)

    if not predictions:
        return pd.DataFrame()

    return pd.DataFrame(predictions).sort_values(["vessel", "date"]).reset_index(drop=True)
