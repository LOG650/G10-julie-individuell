# SARIMA metode

- Modell: `sarima`
- Evalueringsnivå: `flåtenivå`
- Train: `2021-04 til 2024-12`
- Test: `2025-01 til 2026-03`

## Steg 1. Datagrunnlag

Aggregert månedlig flåteserie bygget fra train-settet. Train-serien dekker `2021-04 til 2024-12`.

## Steg 2. Stasjonaritet

Stasjonaritet ble vurdert med ADF-test på originalserie og differensierte serier. Valgt differensiering: `d=0`, `D=0`.

## Steg 3. ACF- og PACF-analyse

ACF og PACF ble brukt på den valgte stasjonære serien for å identifisere kandidatmodeller. Se `acf.png`, `pacf.png` og `kandidatmodeller.md`.

## Steg 4. Modellestimering

Kandidatmodeller ble estimert og rangert med AIC. Valgt modell: `(2, 0, 0)` x `(1, 0, 0, 12)`.

## Steg 5. Modellvalidering

Residualdiagnostikk ble gjennomført med residualplott og Ljung-Box-test, og modellen ble deretter evaluert på holdout-perioden med `MAE=47.7648` og `RMSE=58.3014`.

## Steg 6. Prognose

Endelig prognose for april og mai 2026 ble laget ved å refitte valgt modell på full historikk til og med mars 2026.

## Repo-artefakter

- `stasjonaritet.md`
- `kandidatmodeller.md`
- `acf.png`
- `pacf.png`
- `residualdiagnostikk.png`
