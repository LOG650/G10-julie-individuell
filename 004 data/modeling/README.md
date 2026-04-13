# Modellering

Denne mappen inneholder første versjon av modelleringspipelinen for rapporten.

Målet er å sammenligne fire modeller mot samme problemstilling:

- `SARIMA`
- `Eksponentiell glatting`
- `XGBoost`
- `LSTM`

## Datagrunnlag

Pipelinen leser `004 data/Data som skal brukes Anonymisert.csv` og reshaper den til et long-format med én rad per fartøy og måned.

Det nåværende datasettet har en viktig begrensning:

- reelle observasjoner finnes bare fra april til desember
- dette gir bare 9 tidssteg per fartøy

Derfor er modellenes bruk i denne første versjonen delt i to:

- `Eksponentiell glatting`, `XGBoost` og `LSTM` kan bygges som en første prototype
- `SARIMA` krever mer tidsdybde for en faglig forsvarlig sesongmodell og blir derfor eksplisitt stoppet dersom dataserien er for kort

## Oppsett

Opprett et virtuelt miljø og installer pakkene:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r '004 data/modeling/requirements.txt'
```

## Kjøring

```bash
python '004 data/modeling/run_models.py'
```

Skriptet skriver resultater til `004 data/modeling/results/`.

## Utdata

Skriptet lager:

- `004 data/modeling/results/metrics.json`
- `004 data/modeling/results/predictions.csv`

I tillegg skriver det en kort oppsummering til terminalen om hvilke modeller som ble kjørt, og hvilke som eventuelt ble hoppet over på grunn av datamengde.
