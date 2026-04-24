# 004 data

Denne mappen er delt i tydelige soner for data, kode og genererte artefakter.

## Struktur

- `raw/`: anonymisert masterdatasett
- `processed/`: avledede datasett som `train.csv` og `test.csv`
- `modeling/`: kode for modellpipeline
- `modeling/outputs/`: genererte modellresultater, modellartefakter og logger
- `visualization/`: kode for historiske visualiseringer
- `visualization/outputs/`: genererte figurer, tabeller og sammendrag

## Arbeidsflyt

1. Masterfilen i `raw/` er eneste primærkilde.
2. `modeling/generate_train_test_split.py` bygger eksplisitte datasett i `processed/`.
3. `modeling/run_models.py` skriver samlede og modellspesifikke artefakter til `modeling/outputs/`.
4. `visualization/generate_historical_visuals.py` skriver figurer og tabeller til `visualization/outputs/`.
