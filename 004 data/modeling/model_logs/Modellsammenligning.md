# Modellsammenligning

Dette dokumentet oppsummerer siste kjøring av alle modellene og er ment som arbeidsgrunnlag for metode-, analyse- og diskusjonsdelen i rapporten.

- Sist generert: `2026-04-13T18:48:27`

## Samlet oversikt

| Modell | Status | MAE | RMSE | Kommentar |
| --- | --- | --- | --- | --- |
| Eksponentiell glatting | ok | 2.9111 | 7.7996 | Fartøy brukt: 15 |
| XGBoost | ok | 3.2483 | 7.8304 | Train/Test-rader: 60/30 |
| LSTM | ok | 3.3600 | 8.7522 | Train/Test-sekvenser: 60/30 |
| SARIMA | skipped | ikke tilgjengelig | ikke tilgjengelig | SARIMA ble ikke kjørt fordi dataserien er for kort for en forsvarlig månedlig sesongmodell. Minst 24 observasjoner anbefales. |

## Lenker til detaljfiler

- [SARIMA kode](SARIMA kode.md) og [SARIMA resultater](SARIMA resultater.md)
- [Eksponentiell glatting kode](Eksponentiell glatting kode.md) og [Eksponentiell glatting resultater](Eksponentiell glatting resultater.md)
- [XGBoost kode](XGBoost kode.md) og [XGBoost resultater](XGBoost resultater.md)
- [LSTM kode](LSTM kode.md) og [LSTM resultater](LSTM resultater.md)

## Rangering basert på MAE

1. Eksponentiell glatting med MAE 2.9111 og RMSE 7.7996
2. XGBoost med MAE 3.2483 og RMSE 7.8304
3. LSTM med MAE 3.3600 og RMSE 8.7522

## Hovedfunn

- Lavest MAE i siste kjøring: `Eksponentiell glatting` (2.9111).
- Lavest RMSE i siste kjøring: `Eksponentiell glatting` (7.7996).
- `SARIMA` ble skipped i siste kjøring: SARIMA ble ikke kjørt fordi dataserien er for kort for en forsvarlig månedlig sesongmodell. Minst 24 observasjoner anbefales.
- Resultatene må tolkes som foreløpige fordi datasettet per nå bare dekker 9 måneder.

## Notater Til Rapport

- Bruk tabellen over direkte som grunnlag for sammenligning av modellprestasjon.
- Beskriv at SARIMA ikke kunne evalueres faglig forsvarlig med dagens tidsdybde.
- Diskuter om lav MAE alene er nok, eller om modellens tolkbarhet og datakrav også bør vektlegges.
