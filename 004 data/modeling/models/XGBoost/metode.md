# XGBoost metode

- Modell: `xgboost`
- Evalueringsnivå: `fartøynivå`
- Train: `2021-04 til 2024-12`
- Test: `2025-01 til 2026-03`

## Steg 1. Datagrunnlag

Paneldata med én rad per fartøy og måned, bygget fra train/test-splittet.

## Steg 2. Featureanalyse

Modellen bruker laggede offhire-verdier, rullerende gjennomsnitt og standardavvik, måned som sykliske variabler, fartøy-ID og flagg for spesielle behov.

## Steg 3. Modellspesifikasjon

XGBoost ble satt opp som en gradient boosted tree-modell med et fast feature-set og forhåndsvalgte hyperparametre for dybde, læringsrate og antall trær.

## Steg 4. Modellestimering

Modellen ble trent på train-rader etter feature engineering.

## Steg 5. Modellvalidering

Modellen ble evaluert på holdout-perioden med `MAE=8.5433` og `RMSE=18.3682`.

## Steg 6. Prognose

Prognosen for april og mai 2026 ble generert rekursivt ved å føre prediksjoner tilbake som nye lags.
