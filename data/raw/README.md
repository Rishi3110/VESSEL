# Real data goes here (git-ignored)

Baltic Exchange and Clarksons data are licensed - this folder is ignored by git
except for this file. Drop CSVs here, then in `configs/default.yaml` set:

```yaml
data:
  source: csv
  data_dir: data/raw
```

## Schema

Every file: a `date` column (ISO `YYYY-MM-DD`) plus one or more numeric columns.
Column names must be unique across files. Any frequency works (daily, weekly,
monthly) - everything is resampled to `data.freq` using the **last value
released in each bin**.

**Stamp each row with the date the number refers to** (e.g. month-end for
monthly data) and set `publication_lag_days` in the config to the typical delay
before it becomes public. Do not pre-shift the data yourself.

| File (default name) | Columns used by default config | Typical source | Pub. lag |
|---|---|---|---|
| `rates.csv` | `rate` - the target, $/day or $/t, must be > 0 | Baltic Exchange (BCI 5TC, BPI 82, a route like C5) | 0 |
| `bunker.csv` | `bunker_usd_t` | Ship & Bunker, Platts VLSFO Singapore/Rotterdam | 0 |
| `congestion.csv` | `port_wait_days` | Kpler / Signal / AIS-derived waiting time | 1 |
| `canal.csv` | `panama_restricted` (0/1), add Suez/Red Sea flags as needed | Panama Canal Authority notices | 0 |
| `macro.csv` | `china_pmi`, `china_steel_mt` | NBS / Caixin, worldsteel | 15 |
| `trade.csv` | `iron_ore_imports_mt`, `coal_imports_mt`, `grain_exports_mt` | China customs, USDA, Kpler | 20 |
| `fleet.csv` | `fleet_dwt_m`, `orderbook_dwt_m`, `deliveries_dwt_m`, `scrapping_dwt_m` | Clarksons SIN, BIMCO | 10 |

Want different files or columns? Edit the `data.exog` list and the
`features.*_col` / `features.demand_cols` names in the config - no code change
needed. Remove any domain feature whose input you don't have by setting its
column to `null`.

Run `python -m vessel generate-data --out data/synthetic` to see a complete
example of every file in the expected format.
