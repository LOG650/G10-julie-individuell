# LSTM metode

- Modell: `lstm`
- Evalueringsnivå: `fartøynivå`
- Train: `2021-04 til 2024-12`
- Test: `2025-01 til 2026-03`

## Steg 1. Datagrunnlag

Sekvensdata bygget fra historiske observasjoner per fartøy.

## Steg 2. Sekvensanalyse

Historikken ble omsatt til sekvenser med tre tidligere tidssteg, der både nivå, månedssyklus og spesialkrav inngår som input.

## Steg 3. Modellspesifikasjon

LSTM-modellen ble definert med én LSTM-lag, et tett skjult lag og standardisering av både input og målvariabel.

## Steg 4. Modellestimering

Modellen ble trent på train-sekvenser med tidlig stopping når validering var tilgjengelig.

## Steg 5. Modellvalidering

Holdout-evaluering på testsekvenser ga `MAE=7.7696` og `RMSE=17.1056`.

## Steg 6. Prognose

Prognosen for april og mai 2026 ble generert sekvensielt ved å bruke siste observerte eller predikerte verdier som ny input.
