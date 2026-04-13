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
I tillegg skriver det modellspesifikke dokumenter til `004 data/modeling/model_logs/`.

## Versjonering

For leveransen skal følgende under `004 data/` ligge i Git:

- den anonymiserte CSV-filen som brukes som datagrunnlag
- modelleringskode og avhengigheter i `004 data/modeling/`
- genererte resultater i `004 data/modeling/results/`
- modellspesifikke kode- og resultatlogger i `004 data/modeling/model_logs/`
- en samlet sammenligningsfil i `004 data/modeling/model_logs/Modellsammenligning.md`

Dette gjør at både datagrunnlag, kode og siste kjørte modellresultater kan spores og leveres samlet.

## Utdata

Skriptet lager:

- `004 data/modeling/results/metrics.json`
- `004 data/modeling/results/predictions.csv`
- `004 data/modeling/results/sarima_metrics.json`
- `004 data/modeling/results/exponential_smoothing_metrics.json`
- `004 data/modeling/results/xgboost_metrics.json`
- `004 data/modeling/results/lstm_metrics.json`
- `004 data/modeling/results/<modell>_predictions.csv` for modeller som faktisk produserer prediksjoner
- `004 data/modeling/model_logs/SARIMA kode.md`
- `004 data/modeling/model_logs/SARIMA resultater.md`
- `004 data/modeling/model_logs/Eksponentiell glatting kode.md`
- `004 data/modeling/model_logs/Eksponentiell glatting resultater.md`
- `004 data/modeling/model_logs/XGBoost kode.md`
- `004 data/modeling/model_logs/XGBoost resultater.md`
- `004 data/modeling/model_logs/LSTM kode.md`
- `004 data/modeling/model_logs/LSTM resultater.md`
- `004 data/modeling/model_logs/Modellsammenligning.md`

I tillegg skriver det en kort oppsummering til terminalen om hvilke modeller som ble kjørt, og hvilke som eventuelt ble hoppet over på grunn av datamengde.
