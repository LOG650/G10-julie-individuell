# Historical Data Visualization

Denne mappen inneholder genererte figurer og tabeller for de historiske offhire-dataene i
`004 data/Data som skal brukes Anonymisert.csv`.

## Kjøring

Bruk prosjektets permanente virtuelle miljø:

```bash
source /Users/juliekarlsen/.venvs/g10-julie-individuell/bin/activate
python "004 data/visualization/generate_historical_visuals.py"
```

## Output

- `figures/`: PNG-filer klare for rapport eller vedlegg
- `tables/`: CSV- og Markdown-tabeller
- `summary.md`: kort oversikt over hva som ble generert og hva som er mest relevant for rapporten

## Standardpakke

Skriptet genererer følgende hovedelementer:

- samlet offhire per måned
- heatmap per fartøy og måned
- gjennomsnittlig offhire per fartøy
- boksplott per fartøy
- tidsserie for fartøyene med høyest gjennomsnittlig offhire
- tabeller for datadekning, fartøysammendrag og toppobservasjoner
