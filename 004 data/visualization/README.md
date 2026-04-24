# Historical Data Visualization

Denne mappen inneholder kode for historiske visualiseringer basert på
`004 data/raw/Data som skal brukes Anonymisert.csv`.

## Struktur

- `./generate_historical_visuals.py`: genererer figurer, tabeller og sammendrag
- `outputs/figures/`: PNG-filer klare for rapport eller vedlegg
- `outputs/tables/`: CSV- og Markdown-tabeller
- `outputs/summary.md`: kort oversikt over hva som ble generert

## Kjøring

Bruk prosjektets permanente virtuelle miljø:

```bash
source /Users/juliekarlsen/.venvs/g10-julie-individuell/bin/activate
python "004 data/visualization/generate_historical_visuals.py"
```

## Standardpakke

Skriptet genererer følgende hovedelementer:

- samlet offhire per måned
- heatmap per fartøy og måned
- gjennomsnittlig offhire per fartøy
- boksplott per fartøy
- tidsserie for fartøyene med høyest gjennomsnittlig offhire
- tabeller for datadekning, fartøysammendrag og toppobservasjoner
