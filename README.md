# AI for Safer Roads — Speed Safety Score

Submission for the **AI for Safer Roads Innovation Challenge**, organized by:

- Asian Development Bank (ADB)
- World Bank Development Impact Group
- AI for Good
- International Telecommunication Union (ITU)

**Challenge statement:** How might we use AI and mobility data to determine
where speed limits are misaligned with real-world road conditions,
supporting evidence-based speed management across Asia and the Pacific?

## What this does

Posted speed limits don't always match what's safe for the people actually
using a road — pedestrians, cyclists, and motorcyclists in particular, and
they often become outdated as the surrounding environment changes. Rather
than detecting speeding behaviour, this project asks a different question:
**is the posted speed limit itself appropriate** for each road segment?

It scores ~70,000 road segments across **India (Maharashtra)** and
**Thailand** on how far their posted speed limit is misaligned with Safe
System principles and vulnerable road user (VRU) exposure, combining:

- **How much faster traffic actually drives** than the posted limit (85th
  percentile speed vs. limit).
- **Whether the posted limit itself exceeds Safe System guidance** for that
  road's class (e.g. a 90 km/h limit on a 60 km/h-max secondary road).
- **The likely consequence of a crash at that speed** for an unprotected
  road user (a standard pedestrian fatality-risk curve).
- **How exposed vulnerable road users are** on that segment (urban
  context plus regional helmet-wearing data, as a proxy for two-wheeler
  vulnerability).
- **How many people live along the segment**, derived from 100 m population
  rasters for Maharashtra and Thailand.

The result is a 0–100 **Speed Safety Score** per segment, a High/Medium/Low
risk tier, a **recommended speed limit** (interpolated within the segment's
road-class Safe System range by VRU exposure, rounded to the nearest 10
km/h) and how far the posted limit deviates from it, and an interactive map
you can explore segment by segment — including a one-click link to
street-level imagery for context — surfacing candidate roads for speed
limit review by policymakers.

This is a transparent, hand-weighted scoring formula, not a trained
predictive model — every score is traceable back to the five inputs above.
See `docs/methodology.md` for the full reasoning, the data-quality issues
found along the way, and an honest discussion of where the score's
validation came back weaker than expected.

**Interactive map:** https://augustin-birladeanu.github.io/GeospatialAIBot/
*(requires GitHub Pages enabled on this repo, pointing at `/docs`)*

## Setup

```bash
conda env create -f environment.yml
conda activate ai4saferroads
```

Place the raw GeoJSON exports in `data/raw/` before running anything (not
committed to this repo — see `.gitignore`):

- `data/raw/ADB_Innovation_Maharashtra.geojson`
- `data/raw/ADB_Innovation_Thailand.geojson`
- `data/raw/maharasthra_india_population_data.tif`
- `data/raw/tha_pop_2025_CN_100m_R2025A_v1.tif`

The VRU exposure feature also expects the helmet-wearing survey GeoPackages
under `Archive/` (see Data sources below).

## How to run

Run the notebooks **in order** from the `notebooks/` directory:

1. **`01_data_exploration.ipynb`** — loads both raw files, profiles schemas
   and nulls, checks the segment-ID relationship between the two countries,
   plots sample-size and speed-gap distributions, flags known data
   anomalies.
2. **`02_cleaning_and_features.ipynb`** — filters to reliable segments,
   saves `data/processed/segments_reliable.geojson` and
   `segments_low_confidence.geojson`, builds every score feature.
3. **`03_scoring_and_sensitivity.ipynb`** — computes the Speed Safety Score
   and risk tiers, validates against `RankedPercentile`, runs the ±10%
   weight sensitivity analysis.
4. **`04_exports_and_map.ipynb`** — exports `outputs/segments_scored.geojson`
   / `.gpkg` / `summary_statistics.csv`, and builds the interactive map.

Each notebook hands off intermediate state to the next via small cache files
in `data/processed/` (gitignored, regenerated each run). The reusable logic
itself lives in `src/` (`utils.py`, `features.py`, `score.py`), which the
notebooks import rather than duplicating.

## Repository structure

```
ai4saferroads/
├── data/
│   ├── raw/              # input GeoJSON (not committed)
│   └── processed/        # filtered reliable / low-confidence segments
├── notebooks/             # 01-04, run in order
├── src/
│   ├── utils.py          # loading, schema harmonisation, filtering
│   ├── features.py       # speed_gap, road_mismatch, vru_exposure, bio_risk...
│   ├── score.py           # weighted score, risk tiers, sensitivity analysis
│   └── train_model.py    # unexecuted scaffold for a future predictive model
├── outputs/                # final deliverables (see below)
├── docs/
│   ├── index.html        # interactive map (GitHub Pages source)
│   └── methodology.md    # full methodology and validation discussion
├── Archive/               # source GeoPackages incl. helmet-wearing survey data
└── environment.yml
```

## Output files

| File | Contents |
|---|---|
| `outputs/segments_scored.geojson` | All reliable segments with every engineered feature, the Speed Safety Score, risk tier, and a per-segment `recommended_speed_limit`/`speed_limit_gap` |
| `outputs/segments_scored.gpkg` | Same data as a GeoPackage, for ArcGIS / QGIS use |
| `outputs/summary_statistics.csv` | Segment counts, risk tier breakdown, score/gap statistics, validation correlation, sensitivity overlap |
| `outputs/sample_size_distribution.png` | Histogram of `Sample_Size_Total` across both countries |
| `outputs/speed_gap_distribution.png` | Histogram of 85th-percentile speed minus posted limit |
| `outputs/map.html` / `docs/index.html` | The interactive Folium map (identical files) |
| `data/processed/segments_reliable.geojson` | Segments that passed the reliability filter (pre-feature-engineering) |
| `data/processed/segments_low_confidence.geojson` | Segments excluded by the reliability filter, tagged `low_confidence` |

## Data sources

- **Road network and segments**: [Overture Maps Foundation](https://overturemaps.org/)
  (© OpenStreetMap contributors, Overture Maps Foundation, ODbL licence).
  Thailand accessed December 2024, Maharashtra accessed May 2025.
- **Traffic speed and volume**: [TomTom Move](https://move.tomtom.com/)
  probe data, sampled at ~10km intervals along each road section.
- **Urban/rural classification**: [NASA GRUMP](https://sedac.ciesin.columbia.edu/data/collection/grump-v1)
  (Global Rural Urban Mapping Project), and the `UrbanPC` field for India.
- **Helmet-wearing / motorcycle exposure**: regional survey results provided
  in `Archive/*.gpkg` (Boundaries_4helmet for Maharashtra,
  Thailand_Province_Boundaries for Thailand), used as a VRU exposure proxy.
- **Population exposure**: 2025 100 m population-count GeoTIFFs from
  [WorldPop](https://hub.worldpop.org/) — [Thailand](https://hub.worldpop.org/geodata/summary?id=75663),
  [India](https://hub.worldpop.org/geodata/summary?id=73807) — CC-BY 4.0.
  Bondarenko M., Priyatikanto R., Tejedor-Garavito N., Zhang W., McKeen T.,
  Cunningham A., Woods T., Hilton J., Cihan D., Nosatiuk B., Brinkhoff T.,
  Tatem A., Sorichetta A. 2025. *Constrained estimates of 2015-2030 total
  number of people per grid square at a resolution of 3 arc (~100m at the
  equator), R2025A v1.* Global Demographic Data Project, funded by The Bill
  and Melinda Gates Foundation (INV-045237). WorldPop — School of Geography
  and Environmental Science, University of Southampton.
  DOI:[10.5258/SOTON/WP00839](https://doi.org/10.5258/SOTON/WP00839).
- **2025 Thailand crash records**: Department of Highways accident log
  (`data/raw/Thailandaccident2025.xlsx`), 23,715 geocoded 2025 crash reports
  with fatality/injury counts, used to validate the Speed Safety Score
  against real crash outcomes rather than as a scoring input (Thailand-only
  data — see `docs/methodology.md` for why it isn't folded into the
  weighted score itself). Joined to segments via `src/crashes.py`.
- **Road names**: [Overture Maps Foundation](https://overturemaps.org/)
  (© OpenStreetMap contributors, Overture Maps Foundation, ODbL licence),
  fetched via `scripts/fetch_overture_names.py` and used in
  `src/road_names.py` to backfill segments the raw exports left unnamed,
  resolved to English wherever possible. Supplements, not a replacement
  for, the raw exports' own `names_primary` / `english_ro` fields.
- **Street-level imagery**: [Mapillary](https://www.mapillary.com/), linked
  per segment via `mapillary_url`.
- **Data dictionary**: *"AI for Safer Roads 2026 — Description of data
  sources"*, Richard Owen / Agilysis, May 2026 (PDF included in this repo).

School/market proximity data remains future work.

Full methodology, including how the two countries' differing data schemas
were reconciled and a discussion of the validation results: see
**[`docs/methodology.md`](docs/methodology.md)**.

## Status

Implemented: schema-harmonized loading for both countries, the reliability
filter, all six engineered features, the weighted Speed Safety Score and
risk tiers, sensitivity analysis, GeoJSON/GeoPackage/CSV exports, and the
interactive map. `src/train_model.py` now runs and fits a
`RandomForestRegressor` against `speed_safety_score` itself (a distillation
sanity check, not a model fit to independent outcome data — see that
file's docstring). A genuine outcome-driven model still isn't implemented:
the ADB ATO Road Safety workbook (the crash/fatality data available so
far) is national-level only (one row per country per year), so it can't be
joined to individual road segments as a training label without every
segment in a country collapsing to the same value. A real predictive model
needs segment- or province-level crash data, which hasn't been located yet.

## Future enhancements

- School and market proximity layers to further sharpen local exposure.
- A trained model validated against real outcome data (crash/fatality
  records), rather than the current rule-based score.
- Cross-country model transfer to additional Asia-Pacific countries.
- An interactive web dashboard beyond the static Folium map.

## Team

Augustin Birladeanu

Mounaraga Annavaram

Yassa Ahmad

Gian Fernandez


## Acknowledgements

Developed as part of the AI for Safer Roads Innovation Challenge organized by
the Asian Development Bank in collaboration with the World Bank Development
Impact Group, AI for Good, and the International Telecommunication Union
(ITU).

## License

This repository is intended for research and competition purposes.
