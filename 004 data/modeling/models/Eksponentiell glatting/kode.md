# Eksponentiell glatting kode

Dette dokumentet genereres automatisk fra den aktive implementasjonen i `004 data/modeling/run_models.py`.

- Modell: `exponential_smoothing`
- Funksjon: `run_exponential_smoothing`

```python
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
            "evaluation_level": "fartøynivå",
            "step_2_summary": "Seriene er korte, nulltunge og ujevne, så sesongkomponent ble ikke brukt.",
            "step_3_summary": "Ikke-sesongbasert eksponentiell glatting per fartøy; additiv trend brukes bare når train-serien er lang nok og ikke konstant.",
        },
    )
    return metrics, pred_df
```
