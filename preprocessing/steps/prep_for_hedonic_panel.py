"""Prepare the collapsed parcel panel for residential hedonic modeling.

Collapses the stacked FY2016–FY2025 assessor panel to one row per parcel-year
(the collapsed parcel-year state in ``preprocessing/utils.py``; 
condo unit accounts are aggregated into the parcel, so a condo building is
one observation, not one row per unit). The collapsed panel is then filtered to residential parcels,
retains the hedonic columns, clips obvious outliers, and derives the
price-per-sqft outcome. Enrichment columns (``neighborhood_name``,
``median_hh_income``, ``emp_dist_m``, ``walkability``) are kept when present
— run the corresponding ``add_*_panel.py`` steps first to populate them.

``land_use_code`` in the output is the developer-choice unit-band category
("1 Unit" … "21+ Units", "Land", "Other") built from the derived ``units``
column, so hedonic and nested-logit choice observations sit on the same
partition; the raw assessor code is kept in ``land_use_desc``.

The hedonic precedence variable is SPATIO-TEMPORAL: ``prior_sales_avg`` is
the mean price per sqft of all residential observations within
``ST_LAG_RADIUS_M`` meters (300 m) whose fiscal year is strictly prior to
the observation's — a running neighborhood average that excludes the
current observation, breaking the simultaneous feedback loop described in
the README. A parcel's own earlier years count (they are within the radius
and strictly prior). FY2016 rows have no priors and drop out of the model
input. With the fiscal-year fallback dates, "prior" means earlier fiscal
years; the data has no quarterly transactions, so the time fixed effects
(``y_2017``…``y_2025``) and this lag are both fiscal-year based.

Usage
-----
    python preprocessing/steps/prep_for_hedonic_panel.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from preprocessing.utils import (
    HEDONIC_REP_COLS,
    NO_DEVELOPMENT,
    RESIDENTIAL_LU,
    categorize_units,
    load_or_build_parcel_state,
    require_existing_path,
)

# Land-use category used for a residential parcel with no livable unit count
# (units == 0): it is priced as land, not as a structure.
LAND_CATEGORY = "Land"

PREP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = PREP_DIR.parent / "processed_data" / "assessor_panel.parquet"
DEFAULT_STATE = PREP_DIR.parent / "processed_data" / "parcel_state.parquet"
DEFAULT_OUT = PREP_DIR.parent / "processed_data" / "hedonic_input.parquet"

CORE_NUMERIC = [
    "assessed_total", "assessed_land", "assessed_bldg",
    "living_area", "gross_area", "land_sf",
    "yr_built", "num_floors", "res_units", "units",
]
TIME_COLS = [
    "fy",
] + [f"y_{y}" for y in range(2017, 2026)]
# Spatio-temporal precedence columns, computed in this script.
LAG_COLS = ["prior_sales_avg", "prior_sales_count"]
SPACE_COLS = [
    "geo_pid", "centroid_x", "centroid_y",
    "neighborhood_name", "median_hh_income", "emp_dist_m", "walkability",
]
ATTR_COLS = [
    "land_use_code", "land_use_desc", "bldg_type", "overall_cond",
    "structure_class", "bedrooms", "full_baths", "half_baths",
    "total_rooms", "num_parking", "zip_code",
]


# Radius for the spatio-temporal precedence variable (README spec: 300 m).
ST_LAG_RADIUS_M = 300.0


def add_spatiotemporal_lag(df: pd.DataFrame, price_col: str = "price_per_sqft",
                            radius_m: float = ST_LAG_RADIUS_M) -> pd.DataFrame:
    """Add ``prior_sales_avg`` / ``prior_sales_count``: the mean and count of
    ``price_col`` over all residential observations within ``radius_m`` meters
    whose fiscal year is strictly earlier than the observation's.

    Per year, a cKDTree is built on the pooled centroids of ALL earlier years
    (expanding temporal window), and each current-year row queries its
    neighborhood. The current observation can never appear in its own source
    pool, so there is no contemporaneous leak.
    """
    valid = (
        df[price_col].notna()
        & df["centroid_x"].notna() & df["centroid_y"].notna()
    )
    avg = pd.Series(np.nan, index=df.index, dtype="float64")
    cnt = pd.Series(0, index=df.index, dtype="int64")

    years = sorted(df.loc[valid, "fy"].unique())
    for yr in years:
        src_idx = df.index[(df["fy"] < yr) & valid]
        qry_idx = df.index[(df["fy"] == yr) & valid]
        if len(src_idx) == 0 or len(qry_idx) == 0:
            continue

        tree = cKDTree(df.loc[src_idx, ["centroid_x", "centroid_y"]].to_numpy())
        neighbors = tree.query_ball_point(
            df.loc[qry_idx, ["centroid_x", "centroid_y"]].to_numpy(), r=radius_m,
        )
        counts = np.fromiter((len(n) for n in neighbors), dtype=np.int64,
                             count=len(qry_idx))
        cnt.loc[qry_idx] = counts
        if counts.sum() == 0:
            continue

        prices = df.loc[src_idx, price_col].to_numpy()
        flat = np.concatenate([np.asarray(n, dtype=np.int64) for n in neighbors])
        group = np.repeat(np.arange(len(qry_idx)), counts)
        sums = np.bincount(group, weights=prices[flat], minlength=len(qry_idx))
        avg.loc[qry_idx] = np.where(counts > 0, sums / np.maximum(counts, 1),
                                    np.nan)

    df["prior_sales_avg"] = avg
    df["prior_sales_count"] = cnt
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL,
                        help="Assessor panel parquet (used to build the cache).")
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE,
                        help="Cached collapsed parcel-year state parquet.")
    parser.add_argument("--rebuild", action="store_true",
                        help="Rebuild the collapsed state even if cached.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT,
                        help="Prepared hedonic input parquet.")
    args = parser.parse_args()
    args.output = args.output.expanduser().resolve()
    return args


def prepare_hedonic_panel(state: pd.DataFrame) -> pd.DataFrame:
    # ``state`` is the collapsed parcel-year state (one row per geo_pid, fy),
    # the same collapsed data the devchoice input uses.
    missing = [c for c in HEDONIC_REP_COLS if c not in state.columns]
    if missing:
        raise ValueError(
            f"Parcel-year state is missing hedonic representative columns "
            f"{missing}; rebuild the cache (run with --rebuild or delete "
            f"processed_data/parcel_state.parquet)."
        )
    df = state.reset_index()
    # The model regressor ``land_use_code`` is the unit-band category shared
    # with the devchoice alternatives (categorize_units on the derived units
    # column, with residential units==0 -> Land); the raw assessor code is
    # preserved in ``land_use_desc`` (devchoice keys its price merge on it).
    df["land_use_code"] = (
        categorize_units(df["units"])
        .replace(NO_DEVELOPMENT, LAND_CATEGORY)
        .fillna("Other")
    )
    df = df.loc[df["lu"].isin(RESIDENTIAL_LU)].copy()
    df = df.drop(columns=["lu"], errors="ignore")

    for col in CORE_NUMERIC + ["median_hh_income", "emp_dist_m"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fiscal-year dummies (transaction year; FY base level), re-derived on the
    # collapsed grain. There are no quarterly transactions, so the time fixed
    # effects are yearly.
    yd = pd.get_dummies(df["fy"].astype(int), prefix="y", dtype=int)
    base = f"y_{int(df['fy'].min())}"
    yd = yd.drop(columns=[base], errors="ignore")
    for c in yd.columns:
        df[c] = yd[c].to_numpy()

    # Outcome: price per square foot (README: hedonic predicts $/sqft for
    # developer revenue calculations).
    df["price_per_sqft"] = df["assessed_total"] / df["living_area"].where(
        df["living_area"] > 0
    )

    # Clip obvious outliers, per year so a later boom doesn't clip early years.
    for col, lo_q, hi_q in [
        ("assessed_total", 0.02, 0.98),
        ("land_sf", None, 0.99),
        ("gross_area", None, 0.95),
        ("living_area", None, 0.95),
        ("price_per_sqft", 0.01, 0.99),
        ("median_hh_income", 0.01, None),
    ]:
        if col not in df.columns:
            continue
        if lo_q is not None:
            df[col] = df[col].clip(lower=df.groupby("fy")[col].transform(lambda s: s.quantile(lo_q)))
        if hi_q is not None:
            df[col] = df[col].clip(upper=df.groupby("fy")[col].transform(lambda s: s.quantile(hi_q)))

    df["land_sf"] = df["land_sf"].clip(lower=500)
    df["gross_area"] = df["gross_area"].clip(lower=500)
    df["living_area"] = df["living_area"].clip(lower=500)
    df["num_floors"] = df["num_floors"].clip(lower=1)
    df["yr_built"] = df["yr_built"].clip(upper=2030)

    # Spatio-temporal precedence variable (300 m, strictly prior years) on the
    # clipped outcome, BEFORE dropping rows so the neighbor pool stays maximal.
    print(f"Building spatio-temporal lag (r={ST_LAG_RADIUS_M:.0f} m)...")
    df = add_spatiotemporal_lag(df)

    required = ["assessed_total", "living_area", "land_sf", "yr_built",
                "land_use_code", "geo_pid", "prior_sales_avg"]
    keep = [c for c in (["parcel_id"] + TIME_COLS + LAG_COLS + SPACE_COLS + ATTR_COLS + CORE_NUMERIC
                        + ["price_per_sqft"]) if c in df.columns]
    df = df.dropna(subset=[c for c in required if c in df.columns])
    return df[keep]


def main() -> None:
    args = parse_args()

    state = load_or_build_parcel_state(
        args.panel, args.state, rebuild=args.rebuild,
        extra_rep_cols=HEDONIC_REP_COLS,
    )
    prepared = prepare_hedonic_panel(state)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_parquet(args.output, index=False)
    print(f"Prepared hedonic rows: {len(prepared):,} "
          f"({prepared['geo_pid'].nunique():,} parcels, "
          f"years {sorted(prepared['fy'].unique())})")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
