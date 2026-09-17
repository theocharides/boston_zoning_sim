"""Build the developer-choice dataset: pooled 1-year cross-sections at the
development-footprint grain.

Land-use change is tracked by *location*, not by parcel ID, because PIDs
change when parcels are assembled or subdivided. The unit of observation is
the **development footprint**, not the individual parcel:

- For each choice year t in FY2021–FY2025, year t-1 and year t parcel
  polygons are intersected. A current parcel is a *successor* of a prior
  parcel when the overlap covers at least ``MIN_OVERLAP_SHARE`` of the
  PRIOR parcel's area (a parent-referenced rule, so a large parent that only
  partly overlaps an assembled project still counts).
- A **footprint** is the connected group of parcels linked by those
  successor relations. A lone unchanged parcel is a footprint of one. A
  subdivision (1 prior -> many current) collapses the children back onto the
  parent; an assembly (many prior -> 1 current) collapses the parents into
  the successor's footprint. Either way, "the land that became one
  development project" is a single observation — the thing a developer
  actually decides over.

Footprint state per year is aggregated from the collapsed parcel-year state
(``preprocessing/utils.py``): unit counts, areas, and assessed values are
summed across member parcels; location/context variables (``centroid_*``,
``neighborhood_name``, ``walkability``, ``emp_dist_m``, ``median_hh_income``)
are area-weighted means; the dominant (largest-area) land use is kept.

Outcome (``dev_outcome``): the current unit category of the footprint when
it differs from its prior unit category and the prior land use was
residential; otherwise "No Development". Because the comparison is at the
footprint grain, subdividing a parcel reads as a single transition on the
parent's footprint — not one land-use change per child lot.

State variables carried for each footprint (from its prior-year aggregate):

- ``acquisition_cost`` — prior assessed total value summed over the
  footprint (what the existing buildings implicitly cost the developer)
- ``market_price_psf`` — hedonic market price per sqft by land-use type.
  Filled from ``--hedonic-prices`` (a parquet/CSV with ``land_use_code`` and
  ``market_price_psf``, optionally ``fy``) once the hedonic model produces
  it; NaN until then.
- zoning (``zoning_use``, ``max_far``, ``max_height``, ``max_floors``),
  ``neighborhood_name``, ``land_sf`` (footprint size),
  ``emp_dist_m``, ``walkability``, ``median_hh_income`` — run the
  ``add_*_panel.py`` enrichment steps first
- ``dev_activity_600m`` / ``dev_units_600m`` — spatially lagged development
  activity: count of development footprints (those whose unit count rose
  versus their prior state) and their net new units within 600 m, pooled
  over FY2017–FY2020 — strictly before the choice window (README). Events
  are detected at the same footprint grain as the outcome, so a subdivided
  project counts once. FY2016 cannot produce an event (no FY2015 geometry).

Output: ``processed_data/devchoice_input.parquet``, one row per
(development footprint, choice year).

Usage
-----
    python preprocessing/steps/prep_for_devchoice_spatial.py
    python preprocessing/steps/prep_for_devchoice_spatial.py \
        --hedonic-prices processed_data/hedonic_prices_by_lu.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from preprocessing.utils import (
    NO_DEVELOPMENT,
    RESIDENTIAL_LU,
    STATE_COLS,
    categorize_units,
    load_or_build_parcel_state,
    normalize_land_use,
    require_existing_path,
)

PREP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = PREP_DIR.parent / "processed_data" / "assessor_panel.parquet"
DEFAULT_STATE = PREP_DIR.parent / "processed_data" / "parcel_state.parquet"
GEOMETRY_DIR = PREP_DIR.parent / "processed_data" / "parcel_geometry"
DEFAULT_OUT = PREP_DIR.parent / "processed_data" / "devchoice_input.parquet"
DEFAULT_HEDONIC_PRICES = PREP_DIR.parent / "processed_data" / "hedonic_prices_by_lu.parquet"

METRIC_CRS = "EPSG:26986"
MIN_OVERLAP_SHARE = 0.50

# Choice years: FY2021–FY2025 (needs FY2020 geometry as the first "prior" year).
CHOICE_YEARS = [2021, 2022, 2023, 2024, 2025]

# Spatially lagged development activity (README: strictly before the choice
# window). FY2017 is the first detectable event year (needs FY2016 geometry as
# its prior), so the effective window is FY2017–FY2020, within 600 m.
ACTIVITY_YEARS = [2017, 2018, 2019, 2020]
ACTIVITY_RADIUS_M = 600.0

# Aggregated prior-year parcel state carried as state variables.
PRIOR_STATE_COLS = ["assessed_total", "land_sf"]


def load_year_geometry(fy: int) -> gpd.GeoDataFrame:
    gdf = gpd.read_parquet(GEOMETRY_DIR / f"parcels_fy{fy}.parquet")
    gdf = gdf[["parcel_id", "geometry"]].rename(columns={"parcel_id": "geo_pid"})
    return gdf.drop_duplicates(subset=["geo_pid"]).reset_index(drop=True)


def best_predecessors(cur: gpd.GeoDataFrame, prev: gpd.GeoDataFrame) -> pd.DataFrame:
    """Map each current parcel to its largest-share prior-year predecessor.

    Returns geo_pid, prev_geo_pid, overlap_share for current parcels whose
    best predecessor covers at least MIN_OVERLAP_SHARE of their area.
    """
    cur_m = cur.to_crs(METRIC_CRS)
    prev_m = prev.to_crs(METRIC_CRS)

    pairs = gpd.sjoin(
        cur_m[["geometry"]], prev_m[["geometry"]],
        how="inner", predicate="intersects",
    )
    if pairs.empty:
        return pd.DataFrame(columns=["geo_pid", "prev_geo_pid", "overlap_share"])

    cur_geoms = cur_m.geometry.to_numpy()[pairs.index.to_numpy()]
    prev_geoms = prev_m.geometry.to_numpy()[pairs["index_right"].to_numpy()]
    inter_area = gpd.GeoSeries(
        gpd.GeoSeries(cur_geoms).intersection(gpd.GeoSeries(prev_geoms)),
        crs=METRIC_CRS,
    ).area.to_numpy()
    cur_area = cur_m.geometry.area.to_numpy()[pairs.index.to_numpy()]

    pairs = pairs.assign(
        overlap_share=inter_area / cur_area,
        geo_pid=cur["geo_pid"].to_numpy()[pairs.index.to_numpy()],
        prev_geo_pid=prev["geo_pid"].to_numpy()[pairs["index_right"].to_numpy()],
    )

    best = (
        pairs.sort_values("overlap_share", ascending=False, kind="stable")
        .groupby("geo_pid", sort=True)
        .head(1)
    )
    best = best[best["overlap_share"] >= MIN_OVERLAP_SHARE]
    return best[["geo_pid", "prev_geo_pid", "overlap_share"]].reset_index(drop=True)


def development_events(state: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """Development events per year: parcels whose unit count rose versus their
    spatial predecessor, located at the current-year parcel centroid.

    Assembly note: an assembled parcel compares against its single largest-share
    predecessor, so units on the other absorbed lots are not netted out — the
    metric is a momentum proxy, not an exact unit census.
    """
    events = []
    for fy in years:
        pred = best_predecessors(load_year_geometry(fy), load_year_geometry(fy - 1))
        cur = state.xs(fy, level="fy")[["units", "centroid_x", "centroid_y"]]
        prev = state.xs(fy - 1, level="fy")[["units"]]
        ev = pred.merge(
            cur, left_on="geo_pid", right_index=True, how="left",
        ).merge(
            prev, left_on="prev_geo_pid", right_index=True, how="left",
            suffixes=("", "_prior"),
        )
        ev["units"] = pd.to_numeric(ev["units"], errors="coerce")
        ev["units_prior"] = pd.to_numeric(ev["units_prior"], errors="coerce")
        ev["net_new_units"] = (ev["units"] - ev["units_prior"]).clip(lower=0)
        ev = ev[ev["net_new_units"] > 0].dropna(subset=["centroid_x", "centroid_y"])
        ev["fy"] = fy
        events.append(
            ev[["geo_pid", "fy", "net_new_units", "centroid_x", "centroid_y"]]
        )
        print(f"FY{fy}: {len(events[-1]):,} development events "
              f"({ev['net_new_units'].sum():,.0f} net new units)")
    if not events:
        return pd.DataFrame(
            columns=["geo_pid", "fy", "net_new_units", "centroid_x", "centroid_y"]
        )
    return pd.concat(events, ignore_index=True)


def add_spatial_dev_activity(choice: pd.DataFrame, events: pd.DataFrame,
                             radius_m: float = ACTIVITY_RADIUS_M) -> pd.DataFrame:
    """Spatially lagged development activity: per choice parcel, the count of
    pre-window development events (``dev_activity_600m``) and their net new
    units (``dev_units_600m``) within ``radius_m`` meters of the parcel.

    Event locations are fixed in the pre-choice window, so current developer
    choices cannot retroactively influence them (temporal exogeneity)."""
    choice["dev_activity_600m"] = 0
    choice["dev_units_600m"] = 0.0
    if events.empty:
        return choice

    tree = cKDTree(events[["centroid_x", "centroid_y"]].to_numpy())
    valid = choice["centroid_x"].notna() & choice["centroid_y"].notna()
    qry_idx = choice.index[valid]
    neighbors = tree.query_ball_point(
        choice.loc[qry_idx, ["centroid_x", "centroid_y"]].to_numpy(), r=radius_m,
    )
    counts = np.fromiter((len(n) for n in neighbors), dtype=np.int64,
                         count=len(qry_idx))
    choice.loc[qry_idx, "dev_activity_600m"] = counts
    if counts.sum() > 0:
        flat = np.concatenate([np.asarray(n, dtype=np.int64) for n in neighbors])
        group = np.repeat(np.arange(len(qry_idx)), counts)
        unit_sums = np.bincount(group, weights=events["net_new_units"].to_numpy()[flat],
                                minlength=len(qry_idx))
        choice.loc[qry_idx, "dev_units_600m"] = unit_sums
    return choice


def add_market_price(choice: pd.DataFrame, prices_path: Path | None) -> pd.DataFrame:
    """Merge hedonic market price per sqft by land-use type (prior land use).

    Expected file schema: ``land_use_code`` + ``market_price_psf``, optionally
    ``fy`` for time-varying prices. Until the hedonic model produces this
    file, the column is NaN.
    """
    if not prices_path or not Path(prices_path).exists():
        choice["market_price_psf"] = np.nan
        print("No hedonic price file found — market_price_psf is NaN "
              "(populate once the hedonic model produces prices by LU type).")
        return choice

    prices_path = Path(prices_path)
    prices = (pd.read_parquet(prices_path) if prices_path.suffix == ".parquet"
              else pd.read_csv(prices_path))
    prices = prices.rename(columns={"lu": "land_use_code", "price_per_sqft": "market_price_psf"})
    prices["land_use_code"] = normalize_land_use(prices["land_use_code"])

    keys = ["land_use_code"] + (["fy"] if "fy" in prices.columns else [])
    missing = [c for c in ["market_price_psf"] if c not in prices.columns]
    if missing:
        raise ValueError(f"Hedonic price file {prices_path} is missing columns: {missing}")

    choice = choice.merge(
        prices[keys + ["market_price_psf"]].drop_duplicates(subset=keys),
        left_on=(["prior_lu", "fy"] if "fy" in keys else ["prior_lu"]),
        right_on=keys, how="left",
    ).drop(columns=["land_use_code"], errors="ignore")
    n = choice["market_price_psf"].notna().sum()
    print(f"Market price merged from {prices_path}: {n:,} of {len(choice):,} rows priced")
    return choice


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL,
                        help="Assessor panel parquet (used to build the cache).")
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE,
                        help="Cached collapsed parcel-year state parquet.")
    parser.add_argument("--rebuild", action="store_true",
                        help="Rebuild the collapsed state even if cached.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT,
                        help="Output parquet for the developer choice model.")
    parser.add_argument("--hedonic-prices", type=Path, default=DEFAULT_HEDONIC_PRICES,
                        help="Optional parquet/CSV of hedonic market price per sqft "
                             "by land-use type (columns: land_use_code, market_price_psf, "
                             "optionally fy).")
    args = parser.parse_args()
    args.panel = require_existing_path(args.panel, "Panel parquet")
    args.output = args.output.expanduser().resolve()
    return args


def main() -> None:
    args = parse_args()

    state = load_or_build_parcel_state(args.panel, args.state, rebuild=args.rebuild)
    state["units"] = pd.to_numeric(state["units"], errors="coerce")
    print(f"  {len(state):,} parcel-year states "
          f"({state.index.get_level_values(0).nunique():,} parcels)")

    print(f"\nDetecting pre-window development events (FY{ACTIVITY_YEARS[0]}–"
          f"FY{ACTIVITY_YEARS[-1]})...")
    events = development_events(state, ACTIVITY_YEARS)

    state_vars = [c for c in STATE_COLS if c in state.columns]
    prior_attrs = ["lu", "unit_category",
                   *[c for c in PRIOR_STATE_COLS if c in state.columns],
                   *state_vars]
    frames = []
    for fy in CHOICE_YEARS:
        cur_geo = load_year_geometry(fy)
        prev_geo = load_year_geometry(fy - 1)

        pred = best_predecessors(cur_geo, prev_geo)
        cur_state = state.xs(fy, level="fy")
        prev_state = state.xs(fy - 1, level="fy")

        out = pred.merge(
            cur_state[["lu", "unit_category", "parcel_id"]],
            left_on="geo_pid", right_index=True, how="left",
        ).merge(
            prev_state[prior_attrs],
            left_on="prev_geo_pid", right_index=True, how="left",
            suffixes=("", "_prior"),
        )
        out = out.rename(columns={
            "lu": "current_lu",
            "unit_category": "current_unit_category",
            "lu_prior": "prior_lu",
            "unit_category_prior": "prior_unit_category",
            "assessed_total": "acquisition_cost",
        })
        out["fy"] = fy
        frames.append(out)

        n = len(cur_geo)
        print(
            f"FY{fy}: {len(pred):,} of {n:,} parcels matched to a >= "
            f"{MIN_OVERLAP_SHARE:.0%} predecessor"
        )

    choice = pd.concat(frames, ignore_index=True)

    # Outcome: current unit category when it changed on residential land.
    changed = (
        choice["current_unit_category"].fillna(NO_DEVELOPMENT)
        .ne(choice["prior_unit_category"].fillna(NO_DEVELOPMENT))
    )
    was_residential = choice["prior_lu"].isin(RESIDENTIAL_LU)
    choice["in_choice_set"] = was_residential
    choice["dev_outcome"] = (
        choice["current_unit_category"].where(changed & was_residential)
        .fillna(NO_DEVELOPMENT)
        .astype("string")
    )

    print(f"\nAdding spatially lagged development activity "
          f"({ACTIVITY_RADIUS_M:.0f} m, FY{ACTIVITY_YEARS[0]}–FY{ACTIVITY_YEARS[-1]})...")
    choice = add_spatial_dev_activity(choice, events)
    choice = add_market_price(choice, args.hedonic_prices)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    choice.to_parquet(args.output, index=False)

    print(f"\nChoice observations: {len(choice):,} "
          f"({choice['geo_pid'].nunique():,} parcels, years {CHOICE_YEARS})")
    print("\nOutcome distribution:")
    print(choice["dev_outcome"].value_counts(dropna=False).to_string())
    print("\nDevelopment transitions per year:")
    developed = choice[choice["dev_outcome"] != NO_DEVELOPMENT]
    print(developed["fy"].value_counts().sort_index().to_string())
    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
