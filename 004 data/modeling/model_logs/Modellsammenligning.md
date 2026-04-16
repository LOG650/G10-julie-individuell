# Modellsammenligning

Dette dokumentet oppsummerer siste kjøring av alle modellene og er ment som arbeidsgrunnlag for metode-, analyse- og diskusjonsdelen i rapporten.

- Sist generert: `2026-04-16T12:31:13`
- Evaluering train: `2021-04 til 2024-12`
- Evaluering test: `2025-01 til 2026-03`
- Fremtidsprognoser: `1 måned, 3 måneder, 6 måneder, 12 måneder`
- Prognosevindu: `2026-04` til `2027-03`

## Samlet oversikt

| Modell | Status | MAE | RMSE | sMAPE | Kommentar |
| --- | --- | --- | --- | --- | --- |
| SARIMA | ok | 6.1521 | 16.7823 | 100.4363 | Modellerte fartøy: 15 |
| XGBoost | ok | 7.3487 | 17.4010 | 182.9827 | Walk-forward måneder: 15 |
| LSTM | ok | 7.5722 | 17.4062 | 178.7664 | Sekvenslengde: 12 |
| Eksponentiell glatting | ok | 8.3676 | 17.9544 | 168.6203 | Fartøy brukt: 15 |

## Lenker til detaljfiler

- [SARIMA metode](../models/SARIMA/metode.md), [SARIMA kode](../models/SARIMA/kode.md) og [SARIMA resultater](../models/SARIMA/resultater.md)
- [Eksponentiell glatting metode](../models/Eksponentiell glatting/metode.md), [Eksponentiell glatting kode](../models/Eksponentiell glatting/kode.md) og [Eksponentiell glatting resultater](../models/Eksponentiell glatting/resultater.md)
- [XGBoost metode](../models/XGBoost/metode.md), [XGBoost kode](../models/XGBoost/kode.md) og [XGBoost resultater](../models/XGBoost/resultater.md)
- [LSTM metode](../models/LSTM/metode.md), [LSTM kode](../models/LSTM/kode.md) og [LSTM resultater](../models/LSTM/resultater.md)

## Rangering basert på MAE

1. SARIMA med MAE 6.1521 og RMSE 16.7823
2. XGBoost med MAE 7.3487 og RMSE 17.4010
3. LSTM med MAE 7.5722 og RMSE 17.4062
4. Eksponentiell glatting med MAE 8.3676 og RMSE 17.9544

## Hovedfunn

- Lavest MAE i siste kjøring: `SARIMA` (6.1521).
- Lavest RMSE i siste kjøring: `SARIMA` (16.7823).
- Datasettet dekker observasjoner fra `2021-04` til `2026-03` fordelt på 6 kalenderår.
- Siste år i datasettet er foreløpig ufullstendig og går til `Mars 2026`.

## Notater Til Rapport

- Bruk tabellen over direkte som grunnlag for sammenligning av modellprestasjon.
- Beskriv eksplisitt hvilken tidsdekning datasettet faktisk har, og at siste år kan være ufullstendig.
- Diskuter om lav MAE alene er nok, eller om modellens tolkbarhet og datakrav også bør vektlegges.
