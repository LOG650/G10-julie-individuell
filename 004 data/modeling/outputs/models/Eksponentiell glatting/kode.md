# Eksponentiell glatting kode

Dette dokumentet genereres automatisk fra den aktive implementasjonen i `004 data/modeling/exponential_smoothing.py`.

- Modell: `exponential_smoothing`
- Funksjon: `run_exponential_smoothing`

```python
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
```
