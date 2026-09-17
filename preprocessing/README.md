# Preprocessing Scripts

This folder contains the parcel preprocessing pipeline. It produces **two
separate datasets**:
- Hedonic panel: FY2016–FY2025 (10 years)
- Developer choice: FY2021–FY2025 (5 pooled cross-sections)
- Spatially lagged development activity: FY2017–FY2020 events within 600 m
  (strictly before the choice window; FY2016 cannot produce an event because
  there is no FY2015 geometry to compare against)
- Hedonic spatio-temporal precedence variable: 300 m radius, strictly prior
  fiscal years

### Panel track

The panel track produces **two separate datasets**:

- `processed_data/hedonic_input.parquet` — the **parcel-year panel**
  (FY2016–FY2025, PID-anchored) for the hedonic price model
- `processed_data/devchoice_input.parquet` — the **developer-choice set**:
  five pooled 1-year cross-sections (FY2021–FY2025) where land-use change is
  tracked by *location* (consecutive-year polygon overlay, largest-share
  predecessor >= 50% of parcel area) rather than by PID, so parcel assembly
  and subdivision are handled correctly and not double-counted

Run the whole track with one command:

```bash
python preprocessing/run_panel_prep.py                    # full pipeline
python preprocessing/run_panel_prep.py --skip-income      # no Census API key
python preprocessing/run_panel_prep.py --skip-walkability # no OSM download
python preprocessing/run_panel_prep.py --years 2024 2025  # subset years
```

Individual steps (steps 1–5 read and update `processed_data/assessor_panel.parquet`):

1. `steps/stack_assessors.py`
- Stacks `raw_data/assessors/property-assessment-fyYYYY.csv` (FY2016–FY2025)
  into a parcel-year panel, harmonizing the two assessor schema generations.
- Attaches each year of assessor records to that year's parcel polygons
  (`raw_data/shapes/Parcels__YYYY_.geojson`) using normalized 10-digit PIDs
  (direct match, then condo account → base parcel fallback).
- Adds fiscal-year dummies (`y_2017`…`y_2025`; FY2016 is the base level).
  The data has no quarterly transactions, so the time fixed effects are
  yearly (the transaction year), not quarterly. `fy` plus these dummies fully
  capture time, so no separate observation-date column is kept.
- Outputs:
  - `processed_data/assessor_panel.parquet` — panel with `geo_pid` and
    parcel centroids (`centroid_x`/`centroid_y`, EPSG:26986 meters)
  - `processed_data/parcel_geometry/parcels_fyYYYY.parquet` — per-year
    polygons (GeoParquet, EPSG:4326) for the spatial overlays

```bash
python preprocessing/steps/stack_assessors.py               # all years
python preprocessing/steps/stack_assessors.py --years 2024 2025
```

2. `steps/add_neighborhood_panel.py`
- Spatially joins Boston neighborhood boundaries onto parcel centroids.
- Adds `neighborhood_name` (and `neighborhood_id` if available).
- Joins once per unique `geo_pid`, then broadcasts to all years.
- Requires `preprocessing/raw_data/boston_neighborhood_boundaries.geojson`.

3. `steps/add_income_panel.py`
- Pulls ACS 5-year tract median household income (`B19013_001E`), keyed by
  fiscal year (FY2025 -> ACS 2023, i.e. FY-2).
- Spatially joins parcel centroids to tracts once per `geo_pid` per ACS
  vintage, then broadcasts across the panel.
- Adds `median_hh_income`.
- Requires a Census API key (`CENSUS_API_KEY` env var or `--census-api-key`).

4. `steps/add_employment_dist_panel.py`
- Straight-line distance (meters) from each parcel centroid to the nearest
  employment center/CBD, computed once per `geo_pid`.
- Adds `emp_dist_m`.

5. `steps/add_zoning_panel.py`
- Spatially joins zoning subdistrict polygons onto parcel centroids
  (once per `geo_pid`, then broadcast to all years).
- Adds `zoning_use`, `max_far`, `max_height`, `max_floors`.
- Requires `preprocessing/raw_data/boston_zoning_subdistricts/`.

6. `steps/add_walkability_panel.py`
- Downloads the OSM walking network and amenity destinations (grocery, food,
  education, park, transit) for the parcel extent.
- Scores each unique parcel by shortest-path network distance to the nearest
  amenity per category (linear decay from 100 at 0 m to 0 at 1600 m),
  averaged into one `walkability` score per `geo_pid`.
- Caches per-parcel scores to `processed_data/walkability.parquet` so
  rebuilding the panel does not repeat the OSM download.
- `--sample-size N` scores a random subset for a fast validation run.

7. `steps/prep_for_hedonic_panel.py`
- Reads the shared collapsed parcel-year state, cached at
  `processed_data/parcel_state.parquet` and built once by
  `utils.load_or_build_parcel_state` (one row per `geo_pid` x `fy` — the same
  collapsed data the devchoice input uses). The hedonic prep runs first in
  `run_panel_prep.py` and builds the cache; the devchoice prep reuses it.
  Use `--rebuild` (or delete the cache) after the panel or enrichment steps
  change.
- Filters to residential land uses, clips
  outliers per year, and derives `price_per_sqft` (the hedonic outcome).
- Builds the **spatio-temporal precedence variable**: `prior_sales_avg` =
  mean `price_per_sqft` of all residential observations within **300 m**
  whose fiscal year is strictly prior (a parcel's own earlier years count);
  `prior_sales_count` = number of contributing observations. The current
  observation is never in its own pool, so there is no contemporaneous leak.
  FY2016 rows have no priors and drop out.
- Writes `processed_data/hedonic_input.parquet` (the parcel panel).

8. `steps/prep_for_devchoice_spatial.py`
- Reads the same cached collapsed parcel-year state
  (`processed_data/parcel_state.parquet`; condo logic: a CM row supplies
  attributes and unit count, CD/CP/CC unit rows are aggregated in; RC/A
  units always estimated as `ceil(living_area / 850)`).
- For choice years FY2021–FY2025, intersects each parcel polygon with the
  prior year's polygons and takes the largest-share predecessor (>= 50% of
  area) — so LU change is tracked by location, not PID. Assembled parcels get
  one predecessor; subdivided children each link back to their parent.
- Outcome `dev_outcome`: the current unit category when it changed and the
  prior land use was residential; otherwise "No Development".
- State variables (from the prior-year parcel state): `acquisition_cost`
  (prior assessed total), `land_sf` (parcel size), `zoning_use` + regulation
  limits, `neighborhood_name`, `emp_dist_m`, `walkability`,
  `median_hh_income`, plus zip/condition/transaction-year controls.
- `market_price_psf`: hedonic market price per sqft by land-use type, merged
  from `--hedonic-prices` (parquet/CSV with `land_use_code`,
  `market_price_psf`, optional `fy`) once the hedonic model produces it;
  NaN until then.
- `dev_activity_600m` / `dev_units_600m`: spatially lagged development
  activity — count of development events (parcels whose unit count rose
  versus their spatial predecessor) and their net new units within **600 m**,
  pooled over FY2017–FY2020 (strictly before the choice window).
- Writes `processed_data/devchoice_input.parquet` (pooled 1-year
  cross-sections with `dev_outcome`, prior land use, and prior state
  variables).

### Utilities

- `utils.py` — shared helpers for path checks (`require_existing_path`),
  parcel CSV geometry loading/saving, PID normalization (`normalize_pid`,
  `to_base_pid`), and numeric cleanup. Also holds the shared collapse of
  assessor rows to one parcel-year state per (`geo_pid`, `fy`)
  (`parcel_year_state`) and its cached loader (`load_or_build_parcel_state`,
  which writes/reads `processed_data/parcel_state.parquet`) used by BOTH
  model-input preps, plus helpers `normalize_land_use`, `categorize_units`,
  `RESIDENTIAL_LU`, `STATE_COLS`: condo logic — a CM row supplies
  attributes and unit count, CD/CP/CC unit rows are aggregated in; RC/A
  units always estimated as `ceil(living_area / 850)`.