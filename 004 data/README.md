# 004 data

Denne mappen er delt i tydelige soner for data, kode og genererte artefakter.

## Struktur

- `raw/`: anonymisert masterdatasett
- `processed/`: avledede datasett som `train.csv` og `test.csv`
- `modeling/`: kode for modellpipeline
- `modeling/outputs/`: genererte modellresultater, modellartefakter og logger
- `visualization/`: kode for historiske visualiseringer
- `visualization/outputs/`: genererte figurer, tabeller og sammendrag

## Rask navigering

- Datagrunnlaget ligger i `raw/`.
- Train/test-filene som brukes i modelleringen ligger i `processed/`.
- Alle fire modellene ligger i `modeling/`.
- Samlede modellresultater ligger i `modeling/outputs/shared/`.
- Resultater for hver enkelt modell ligger i `modeling/outputs/models/`.
- Historiske figurer og tabeller som ikke er modellspesifikke ligger i `visualization/outputs/`.

## Modelloversikt

| Modell | Kode | Resultater |
| --- | --- | --- |
| SARIMA | `modeling/sarima.py` | `modeling/outputs/models/SARIMA/` |
| Eksponentiell glatting | `modeling/exponential_smoothing.py` | `modeling/outputs/models/Eksponentiell glatting/` |
| XGBoost | `modeling/xgboost_model.py` | `modeling/outputs/models/XGBoost/` |
| LSTM | `modeling/lstm_model.py` | `modeling/outputs/models/LSTM/` |

## Arbeidsflyt

1. Masterfilen i `raw/` er eneste primærkilde.
2. `modeling/generate_train_test_split.py` bygger eksplisitte datasett i `processed/`.
3. `modeling/run_models.py` skriver samlede og modellspesifikke artefakter til `modeling/outputs/`.
4. `visualization/generate_historical_visuals.py` skriver figurer og tabeller til `visualization/outputs/`.
