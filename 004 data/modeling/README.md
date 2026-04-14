# Modellering

Denne mappen inneholder første versjon av modelleringspipelinen for rapporten.

Målet er å sammenligne fire modeller mot samme problemstilling:

- `SARIMA`
- `Eksponentiell glatting`
- `XGBoost`
- `LSTM`

## Datagrunnlag

Pipelinen leser `004 data/Data som skal brukes Anonymisert.csv` og reshaper den til et long-format med én rad per fartøy og måned.

Det nåværende datasettet dekker månedlige observasjoner fra april 2021 til mars 2026.
Det gir normalt inntil 60 tidssteg per fartøy, men med to praktiske begrensninger:

- 2021 starter i april fordi dataserien dekker de siste fem årene
- 2026 er foreløpig bare observert til og med mars

Pipelinen brukes derfor til to ting:

- historisk evaluering av modellene på de siste observerte månedene
- prognoser for april og mai 2026 basert på alle tilgjengelige observasjoner til og med mars 2026

## Oppsett

For å unngå støy i Git og VS Code bør det virtuelle miljøet ligge utenfor repoet.
På denne maskinen brukes den permanente stien
`/Users/juliekarlsen/.venvs/g10-julie-individuell`.

Hvis miljøet allerede finnes, aktiver det slik:

```bash
source /Users/juliekarlsen/.venvs/g10-julie-individuell/bin/activate
```

Hvis miljøet må opprettes på nytt, bruk:

```bash
python3 -m venv /Users/juliekarlsen/.venvs/g10-julie-individuell
source /Users/juliekarlsen/.venvs/g10-julie-individuell/bin/activate
pip install -r '004 data/modeling/requirements.txt'
```

## Kjøring

I denne fasen kjøres bare `Eksponentiell glatting` fullt ut i standardskriptet.
De andre modellene beholdes i koden, men brukes ikke i standardkjøringen før vi eksplisitt velger dem senere.

```bash
python '004 data/modeling/run_models.py'
```

Skriptet skriver resultater til `004 data/modeling/results/`.
I tillegg skriver det modellspesifikke filer til egne mapper under `004 data/modeling/models/`.

## Versjonering

For leveransen skal følgende under `004 data/` ligge i Git:

- den anonymiserte CSV-filen som brukes som datagrunnlag
- modelleringskode og avhengigheter i `004 data/modeling/`
- genererte resultater i `004 data/modeling/results/`
- modellspesifikke kode- og resultatfiler i `004 data/modeling/models/<modell>/`
- en samlet sammenligningsfil i `004 data/modeling/model_logs/Modellsammenligning.md`

Dette gjør at både datagrunnlag, kode og siste kjørte modellresultater kan spores og leveres samlet.

## Utdata

Skriptet lager:

- `004 data/modeling/results/metrics.json`
- `004 data/modeling/results/predictions.csv`
- `004 data/modeling/results/future_predictions.csv`
- `004 data/modeling/models/<modell>/metrics.json`
- `004 data/modeling/models/<modell>/predictions.csv`
- `004 data/modeling/models/<modell>/future_predictions.csv`
- `004 data/modeling/models/<modell>/kode.md`
- `004 data/modeling/models/<modell>/resultater.md`
- `004 data/modeling/model_logs/Modellsammenligning.md`

I tillegg skriver det en kort oppsummering til terminalen om hvilke modeller som ble kjørt, og hvilke som eventuelt ble hoppet over på grunn av datamengde.
