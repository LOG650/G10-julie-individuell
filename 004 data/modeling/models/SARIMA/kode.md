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
```
