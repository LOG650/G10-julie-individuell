# Modellsammenligning

Dette dokumentet oppsummerer siste kjøring av alle modellene og er ment som arbeidsgrunnlag for metode-, analyse- og diskusjonsdelen i rapporten.

- Sist generert: `2026-04-14T17:52:50`
- Evaluering train: `2021-04 til 2024-12`
- Evaluering test: `2025-01 til 2026-03`
- Fremtidsprognoser trener på full historikk: `2021-04 til 2026-03`

## Samlet oversikt

| Modell | Status | MAE | RMSE | Kommentar |
| --- | --- | --- | --- | --- |
| Eksponentiell glatting | ok | 6.7212 | 17.5192 | Fartøy brukt: 15 |

## Lenker til detaljfiler

- [Eksponentiell glatting kode](../models/Eksponentiell glatting/kode.md) og [Eksponentiell glatting resultater](../models/Eksponentiell glatting/resultater.md)

## Rangering basert på MAE

1. Eksponentiell glatting med MAE 6.7212 og RMSE 17.5192

## Hovedfunn

- Lavest MAE i siste kjøring: `Eksponentiell glatting` (6.7212).
- Lavest RMSE i siste kjøring: `Eksponentiell glatting` (17.5192).

## Fremtidsprognoser

- Prognoser for neste måneder er lagret i `004 data/modeling/results/future_predictions.csv` for datoene `2026-04-01`, `2026-05-01`.
- Datasettet dekker observasjoner fra `2021-04` til `2026-03` fordelt på 6 kalenderår.
- Siste år i datasettet er foreløpig ufullstendig og går til `Mars 2026`.

## Notater Til Rapport

- Bruk tabellen over direkte som grunnlag for sammenligning av modellprestasjon.
- Beskriv eksplisitt hvilken tidsdekning datasettet faktisk har, og at siste år kan være ufullstendig.
- Diskuter om lav MAE alene er nok, eller om modellens tolkbarhet og datakrav også bør vektlegges.
