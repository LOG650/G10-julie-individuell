# Modellsammenligning

Dette dokumentet oppsummerer siste kjøring av alle modellene og er ment som arbeidsgrunnlag for metode-, analyse- og diskusjonsdelen i rapporten.

- Sist generert: `2026-04-14T18:53:23`
- Evaluering train: `2021-04 til 2024-12`
- Evaluering test: `2025-01 til 2026-03`
- Fremtidsprognoser trener på full historikk: `2021-04 til 2026-03`

## Samlet oversikt

| Modell | Status | MAE | RMSE | Kommentar |
| --- | --- | --- | --- | --- |
| Eksponentiell glatting | ok | 6.7212 | 17.5192 | Fartøy brukt: 15 |
| LSTM | ok | 7.7696 | 17.1056 | Train/Test-sekvenser: 630/225 |
| XGBoost | ok | 8.5433 | 18.3682 | Train/Test-rader: 630/225 |
| SARIMA | ok | 47.7648 | 58.3014 | Flåtenivå | (2, 0, 0) x (1, 0, 0, 12) |

## Lenker til detaljfiler

- [SARIMA metode](../models/SARIMA/metode.md), [SARIMA kode](../models/SARIMA/kode.md) og [SARIMA resultater](../models/SARIMA/resultater.md)
- [Eksponentiell glatting metode](../models/Eksponentiell glatting/metode.md), [Eksponentiell glatting kode](../models/Eksponentiell glatting/kode.md) og [Eksponentiell glatting resultater](../models/Eksponentiell glatting/resultater.md)
- [XGBoost metode](../models/XGBoost/metode.md), [XGBoost kode](../models/XGBoost/kode.md) og [XGBoost resultater](../models/XGBoost/resultater.md)
- [LSTM metode](../models/LSTM/metode.md), [LSTM kode](../models/LSTM/kode.md) og [LSTM resultater](../models/LSTM/resultater.md)

## Rangering basert på MAE

1. Eksponentiell glatting med MAE 6.7212 og RMSE 17.5192
2. LSTM med MAE 7.7696 og RMSE 17.1056
3. XGBoost med MAE 8.5433 og RMSE 18.3682
4. SARIMA med MAE 47.7648 og RMSE 58.3014

## Hovedfunn

- Lavest MAE i siste kjøring: `Eksponentiell glatting` (6.7212).
- Lavest RMSE i siste kjøring: `LSTM` (17.1056).

## Fremtidsprognoser

- Prognoser for neste måneder er lagret i `004 data/modeling/results/future_predictions.csv` for datoene `2026-04-01`, `2026-05-01`.
- Datasettet dekker observasjoner fra `2021-04` til `2026-03` fordelt på 6 kalenderår.
- Siste år i datasettet er foreløpig ufullstendig og går til `Mars 2026`.

## Notater Til Rapport

- Bruk tabellen over direkte som grunnlag for sammenligning av modellprestasjon.
- Beskriv eksplisitt hvilken tidsdekning datasettet faktisk har, og at siste år kan være ufullstendig.
- Diskuter om lav MAE alene er nok, eller om modellens tolkbarhet og datakrav også bør vektlegges.
