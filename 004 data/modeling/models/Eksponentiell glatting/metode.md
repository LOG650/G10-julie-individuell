# Eksponentiell glatting metode

- Modell: `exponential_smoothing`
- Evalueringsnivå: `fartøynivå`
- Train: `2021-04 til 2024-12`
- Test: `2025-01 til 2026-03`

## Steg 1. Datagrunnlag

Egne månedlige tidsserier per fartøy bygget fra train-settet.

## Steg 2. Mønsteranalyse

Seriene ble vurdert som korte, nulltunge og ujevne. Det ble derfor ikke brukt egen sesongkomponent i modellen.

## Steg 3. Modellspesifikasjon

Det ble brukt ikke-sesongbasert eksponentiell glatting per fartøy. Additiv trend ble bare aktivert når train-serien hadde minst 5 observasjoner og mer enn én unik verdi.

## Steg 4. Modellestimering

Hver fartøysserie ble estimert separat på train-settet.

## Steg 5. Modellvalidering

Modellen ble evaluert med ekspanderende 1-stegs prognoser gjennom testperioden. Resultat: `MAE=6.7212` og `RMSE=17.5192`.

## Steg 6. Prognose

Prognose for april og mai 2026 ble laget per fartøy på full historikk til og med mars 2026.
