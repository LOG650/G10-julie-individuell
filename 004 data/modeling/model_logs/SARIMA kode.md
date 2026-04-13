# SARIMA kode

Dette dokumentet genereres automatisk fra den aktive implementasjonen i `004 data/modeling/run_models.py`.

- Modell: `sarima`
- Funksjon: `run_sarima`

```python
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
```
