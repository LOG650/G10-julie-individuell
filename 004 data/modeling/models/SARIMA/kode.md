# SARIMA kode

Dette dokumentet genereres automatisk fra den aktive implementasjonen i `004 data/modeling/run_models.py`.

- Modell: `sarima`
- Funksjon: `run_sarima`

```python
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

    selected_d, selected_D, transformed_series, stationarity_results = select_sarima_differencing(
        train_series
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
    save_sarima_diagnostic_plots(train_series, transformed_series, residuals)
    write_dataframe_artifacts(
        pd.DataFrame(stationarity_results),
        model_artifact_path("sarima", "stasjonaritet.csv"),
        "SARIMA stasjonaritet",
    )
    write_dataframe_artifacts(
        candidate_df.head(10),
        model_artifact_path("sarima", "kandidatmodeller.csv"),
        "SARIMA kandidatmodeller",
    )

    history = train_series.copy()
    prediction_rows: list[dict[str, Any]] = []
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
            "evaluation_level": "flåtenivå",
            "selected_d": selected_d,
            "selected_D": selected_D,
            "selected_order": {
                "p": selected_order[0],
                "d": selected_order[1],
                "q": selected_order[2],
            },
            "selected_seasonal_order": {
                "P": selected_seasonal_order[0],
                "D": selected_seasonal_order[1],
                "Q": selected_seasonal_order[2],
            },
            "selected_aic": float(best_candidate["aic"]),
            "selected_bic": float(best_candidate["bic"]),
            "ljung_box_lag": int(ljung_lag),
            "ljung_box_pvalue": float(ljung_box["lb_pvalue"].iloc[0]),
            "stationarity_results": stationarity_results,
            "candidate_models": candidate_df.head(10).to_dict(orient="records"),
            "artifact_files": {
                "stasjonaritet": "stasjonaritet.md",
                "kandidatmodeller": "kandidatmodeller.md",
                "acf": "acf.png",
                "pacf": "pacf.png",
                "residualdiagnostikk": "residualdiagnostikk.png",
            },
        },
    )
    return metrics, pred_df
```
