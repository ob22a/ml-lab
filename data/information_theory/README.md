# Information Theory Laboratory data

## Multilingual UDHR snapshot

`udhr_eng.txt`, `udhr_amh.txt`, and `udhr_spa.txt` are static UTF-8 snapshots
of the English, Amharic, and Spanish Universal Declaration of Human Rights
translations from [`wooorm/udhr`](https://github.com/wooorm/udhr), downloaded
2026-09-13. The packaging repository is MIT licensed; the underlying
translations originate from OHCHR and the Unicode UDHR project. Retain source
attribution when reproducing them and remove only the first language identifier word. 
The notebook analyzes character statistics; it does not treat one document as representative of an entire language.

## Transfermarkt-derived football snapshot

`football_players_snapshot.csv` is a deterministic 12,000-player sample built
2026-09-13 from the open
[`dcaribou/transfermarkt-datasets`](https://github.com/dcaribou/transfermarkt-datasets)
tables `players.csv.gz` and `appearances.csv.gz`.

- Target: current `market_value_in_eur` from the players table.
- Performance fields: appearances, goals, assists, minutes, and cards summed
  over all appearances in the source snapshot.
- Age: calculated from date of birth at 2026-09-13.
- Filters: positive market value, age 15–45.
- Sampling: if more than 12,000 eligible rows exist, pandas
  `sample(12000, random_state=42)` after sorting by `player_id`.

The snapshot is bundled so Notebook 06 does not depend on a live API. Values
are observational and Transfermarkt-derived; mutual information indicates
dependence, not causality.

Notebook 06 filters this general snapshot to `Centre-Forward`,
`Second Striker`, `Left Winger`, `Right Winger`, and `Attacking Midfield`
before analyzing goals, assists, and market value. Goalkeepers and defenders
are excluded because the source appearances table has no clean-sheet or
goals-conceded metric with which to evaluate their role fairly.
