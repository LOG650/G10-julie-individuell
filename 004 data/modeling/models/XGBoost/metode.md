# XGBoost metode

- Modell: `xgboost`
- Evalueringsnivå: `fartøynivå`
- Train: `2021-04 til 2024-12`
- Test: `2025-01 til 2026-03`

## Steg 1. Datagrunnlag

Paneldata med én rad per fartøy og måned, bygget fra train/test-splittet.

## Steg 2. Featureanalyse

Modellen bruker laggede offhire-verdier, rullerende statistikk, kalendervariabler, fartøy-ID og flagg for spesielle behov.

## Steg 3. Modellspesifikasjon

XGBoost ble satt opp som en gradient boosted tree-modell med et fast feature-set og forhåndsvalgte hyperparametre for dybde, læringsrate og antall trær.

## Steg 4. Modellestimering

Modellen ble re-trent måned for måned i en ekspanderende 1-stegs evaluering.

## Steg 5. Modellvalidering

Modellen ble evaluert på testperioden med `MAE=7.3487`, `RMSE=17.4010` og `sMAPE=182.9827`.

## Repo-artefakter

- `feature_importance.md`
- `feature_importance.png`
- `representativ_testplot.png`
