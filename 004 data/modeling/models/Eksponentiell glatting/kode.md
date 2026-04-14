# Eksponentiell glatting kode

Dette dokumentet genereres automatisk fra den aktive implementasjonen i `004 data/modeling/run_models.py`.

- Modell: `exponential_smoothing`
- Funksjon: `run_exponential_smoothing`

```python
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
        forecast = fit_or_fallback_exponential_forecast(train_series, len(test_df))

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
```
