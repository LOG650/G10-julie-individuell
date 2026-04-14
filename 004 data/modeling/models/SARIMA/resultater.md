# SARIMA resultater

Dette dokumentet genereres automatisk ved hver kjøring av `004 data/modeling/run_models.py`.

- Sist generert: `2026-04-14T18:53:23`
- Status: `ok`
- MAE: `47.7648`
- RMSE: `58.3014`
- Evaluering train: `2021-04 til 2024-12`
- Evaluering test: `2025-01 til 2026-03`
- Fremtidsprognoser trener på full historikk: `2021-04 til 2026-03`

## Detaljer

```json
{
  "evaluation_train_period": "2021-04 til 2024-12",
  "evaluation_test_period": "2025-01 til 2026-03",
  "forecast_training_period": "2021-04 til 2026-03",
  "train_observations": 675,
  "test_observations": 227,
  "forecast_observations": 902,
  "series_type": "fleet_total",
  "test_rows": 15,
  "evaluation_method": "ekspanderende 1-stegs prognose på aggregert flåteserie",
  "evaluation_level": "flåtenivå",
  "selected_d": 0,
  "selected_D": 0,
  "selected_order": {
    "p": 2,
    "d": 0,
    "q": 0
  },
  "selected_seasonal_order": {
    "P": 1,
    "D": 0,
    "Q": 0
  },
  "selected_aic": 349.44167838734666,
  "selected_bic": 355.17762720528725,
  "ljung_box_lag": 12,
  "ljung_box_pvalue": 0.739203321606996,
  "stationarity_results": [
    {
      "label": "Ingen differensiering",
      "d": 0,
      "D": 0,
      "n_obs": 45,
      "adf_stat": -4.052867517471841,
      "p_value": 0.0011576269934062248,
      "stationary": true,
      "critical_5pct": -2.929885661157025
    },
    {
      "label": "Første differense",
      "d": 1,
      "D": 0,
      "n_obs": 44,
      "adf_stat": -9.47072071590908,
      "p_value": 4.113336882246213e-16,
      "stationary": true,
      "critical_5pct": -2.931549768951162
    },
    {
      "label": "Sesongdifferense (12)",
      "d": 0,
      "D": 1,
      "n_obs": 33,
      "adf_stat": -2.3264105470308,
      "p_value": 0.16359145515560763,
      "stationary": false,
      "critical_5pct": -2.960525341210433
    },
    {
      "label": "Første + sesongdifferense",
      "d": 1,
      "D": 1,
      "n_obs": 32,
      "adf_stat": -0.533452997899741,
      "p_value": 0.8853214023938968,
      "stationary": false,
      "critical_5pct": -3.013097747543462
    }
  ],
  "candidate_models": [
    {
      "p": 2,
      "d": 0,
      "q": 0,
      "P": 1,
      "D": 0,
      "Q": 0,
      "s": 12,
      "aic": 349.44167838734666,
      "bic": 355.17762720528725
    },
    {
      "p": 2,
      "d": 0,
      "q": 1,
      "P": 1,
      "D": 0,
      "Q": 0,
      "s": 12,
      "aic": 350.4576752371419,
      "bic": 357.62761125956763
    },
    {
      "p": 2,
      "d": 0,
      "q": 2,
      "P": 1,
      "D": 0,
      "Q": 0,
      "s": 12,
      "aic": 352.1122420731909,
      "bic": 360.71616530010175
    },
    {
      "p": 1,
      "d": 0,
      "q": 1,
      "P": 1,
      "D": 0,
      "Q": 0,
      "s": 12,
      "aic": 360.3029672915904,
      "bic": 366.1659109027893
    },
    {
      "p": 1,
      "d": 0,
      "q": 2,
      "P": 1,
      "D": 0,
      "Q": 0,
      "s": 12,
      "aic": 362.07538561756604,
      "bic": 369.40406513156466
    },
    {
      "p": 1,
      "d": 0,
      "q": 0,
      "P": 1,
      "D": 0,
      "Q": 0,
      "s": 12,
      "aic": 366.25866267438727,
      "bic": 370.6558703827865
    },
    {
      "p": 0,
      "d": 0,
      "q": 2,
      "P": 1,
      "D": 0,
      "Q": 0,
      "s": 12,
      "aic": 379.01818167168693,
      "bic": 385.0042119175529
    },
    {
      "p": 0,
      "d": 0,
      "q": 1,
      "P": 1,
      "D": 0,
      "Q": 0,
      "s": 12,
      "aic": 383.31426412927397,
      "bic": 387.8037868136734
    },
    {
      "p": 0,
      "d": 0,
      "q": 0,
      "P": 1,
      "D": 0,
      "Q": 0,
      "s": 12,
      "aic": 384.15779213441664,
      "bic": 387.1508072573496
    },
    {
      "p": 1,
      "d": 0,
      "q": 2,
      "P": 0,
      "D": 0,
      "Q": 0,
      "s": 12,
      "aic": 475.83896372499674,
      "bic": 482.7896421981302
    }
  ],
  "artifact_files": {
    "stasjonaritet": "stasjonaritet.md",
    "kandidatmodeller": "kandidatmodeller.md",
    "acf": "acf.png",
    "pacf": "pacf.png",
    "residualdiagnostikk": "residualdiagnostikk.png"
  }
}
```

## Prediksjoner

| model | vessel | date | actual | prediction |
| --- | --- | --- | --- | --- |
| sarima | fleet_total | 2025-01-01 | 93.8900 | 63.2536 |
| sarima | fleet_total | 2025-02-01 | 56.0000 | 79.4994 |
| sarima | fleet_total | 2025-03-01 | 61.4500 | 64.8068 |
| sarima | fleet_total | 2025-04-01 | 111.8800 | 48.9838 |
| sarima | fleet_total | 2025-05-01 | 1.7900 | 74.6953 |
| sarima | fleet_total | 2025-06-01 | 43.7400 | 52.7479 |
| sarima | fleet_total | 2025-07-01 | 115.4100 | 17.8026 |
| sarima | fleet_total | 2025-08-01 | 29.0000 | 68.1950 |
| sarima | fleet_total | 2025-09-01 | 40.4500 | 62.2311 |
| sarima | fleet_total | 2025-10-01 | 152.0000 | 29.0852 |

## Fremtidsprognoser

| model | vessel | date | prediction |
| --- | --- | --- | --- |
| sarima | fleet_total | 2026-04-01 | 71.4954 |
| sarima | fleet_total | 2026-05-01 | 71.7242 |

Viser de første 10 av totalt 15 prediksjonsrader.
