# LSTM metode

- Modell: `lstm`
- Evalueringsnivå: `fartøynivå`
- Train: `2021-04 til 2024-12`
- Test: `2025-01 til 2026-03`

## Steg 1. Datagrunnlag

Sekvensdata bygget fra historiske observasjoner per fartøy.

## Steg 2. Sekvensanalyse

Historikken ble omsatt til sekvenser med tolv tidligere tidssteg, der både nivå, månedssyklus og spesialkrav inngår som input.

## Steg 3. Modellspesifikasjon

LSTM-modellen ble definert med ett LSTM-lag, ett tett lag og standardisering av både input og målvariabel basert på treningsdata.

## Steg 4. Modellestimering

Modellen ble re-trent måned for måned i en ekspanderende 1-stegs evaluering.

## Steg 5. Modellvalidering

Holdout-evaluering på testsekvenser ga `MAE=7.5722`, `RMSE=17.4062` og `sMAPE=178.7664`.

## Repo-artefakter

- `training_history.md`
- `training_history.png`
- `representativ_testplot.png`
