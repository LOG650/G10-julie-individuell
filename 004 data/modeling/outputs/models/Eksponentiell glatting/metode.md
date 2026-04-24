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

Det ble sammenlignet et begrenset sett av additive ETS-varianter per fartøy. Valgt spesifikasjon per fartøy er lagret i `modellvalg_per_fartoy.md`.

## Steg 4. Modellestimering

Hver fartøysserie ble estimert separat på train-settet, og beste ETS-variant ble beholdt.

## Steg 5. Modellvalidering

Modellen ble evaluert med ekspanderende 1-stegs prognoser gjennom testperioden. Resultat: `MAE=8.3676`, `RMSE=17.9544` og `sMAPE=168.6203`.

## Repo-artefakter

- `modellvalg_per_fartoy.md`
- `residualdiagnostikk.md`
- `representativ_testplot.png`
