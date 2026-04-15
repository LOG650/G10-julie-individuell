# SARIMA metode

- Modell: `sarima`
- Evalueringsnivå: `fartøynivå`
- Train: `2021-04 til 2024-12`
- Test: `2025-01 til 2026-03`

## Steg 1. Datagrunnlag

Månedlige tidsserier ble bygget per fartøy fra train-settet. Train-seriene dekker `2021-04 til 2024-12`.

## Steg 2. Stasjonaritet

ADF-test ble brukt som støtte for differensieringsvalg per fartøy. Endelig modellvalg ble deretter gjort gjennom et begrenset ARIMA/SARIMA-søk.

## Steg 3. ACF- og PACF-analyse

ACF og PACF ble lagret for et representativt fartøy som diagnostisk støtte. Se `acf.png`, `pacf.png` og `kandidatmodeller.md`.

## Steg 4. Modellestimering

Kandidatmodeller ble estimert per fartøy og rangert med AIC, BIC og parsimoni. Valgte spesifikasjoner er oppsummert i `modellvalg_per_fartoy.md`.

## Steg 5. Modellvalidering

Residualdiagnostikk ble gjennomført per fartøy med Ljung-Box-test, og modellene ble evaluert med ekspanderende 1-stegs prognoser på testperioden. Samlet resultat: `MAE=6.1521`, `RMSE=16.7823` og `sMAPE=100.4363`.

## Repo-artefakter

- `stasjonaritet.md`
- `kandidatmodeller.md`
- `modellvalg_per_fartoy.md`
- `residualdiagnostikk.md`
- `acf.png`
- `pacf.png`
- `residualdiagnostikk.png`
- `representativ_testplot.png`
