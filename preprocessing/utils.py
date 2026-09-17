"""Shared utilities for preprocessing pipeline."""

from __future__ import annotations

from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import wkt

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))


def require_existing_path(path: Path, label: str = "Path") -> Path:
    """Resolve ``path`` and raise a clear error if it does not exist."""
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def load_parcels_csv(csv_path: Path, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """Load parcels CSV with WKT geometry column and return as GeoDataFrame."""
    df = pd.read_csv(csv_path, low_memory=False)
    if "geometry" not in df.columns:
        raise ValueError("Input CSV must contain a 'geometry' WKT column.")

    geoms = df["geometry"].map(
        lambda value: wkt.loads(value) if isinstance(value, str) and value.strip() else None
    )
    return gpd.GeoDataFrame(df, geometry=geoms, crs=crs)


def save_parcels_csv(gdf: gpd.GeoDataFrame, output_path: Path) -> None:
    """Save GeoDataFrame to CSV with geometry as WKT."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(gdf.drop(columns=["geometry"], errors="ignore"))
    df["geometry"] = gdf.geometry.map(
        lambda geom: geom.wkt if getattr(geom, "wkt", None) is not None else pd.NA
    )
    df.to_csv(output_path, index=False)


def to_point_geometry(geom):
    """Convert any geometry to a point for distance calculations."""
    if geom is None:
        return None
    return geom if geom.geom_type == "Point" else geom.representative_point()


def normalize_pid(series: pd.Series) -> pd.Series:
    """Normalize parcel/account IDs to a 10-digit numeric string."""
    return series.astype(str).str.replace(r"\D", "", regex=True).str.zfill(10)


def to_base_pid(pid_norm: pd.Series) -> pd.Series:
    """Convert an account PID to a likely base parcel PID."""
    return pid_norm.str.slice(0, 7) + "000"


def clean_numeric(series: pd.Series) -> pd.Series:
    """Parse numeric-looking fields that may include commas or symbols."""
    as_text = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip()
    )
    as_text = as_text.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return pd.to_numeric(as_text, errors="coerce")


def clean_year_series(series: pd.Series) -> pd.Series:
    """Parse year fields and correct obvious single-digit suffix typos.

    Known source correction: 20198 -> 2019.
    """
    raw_text = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip()
    )
    raw_text = raw_text.replace({"": np.nan, "nan": np.nan, "None": np.nan, "20198": "2019"})
    years = pd.to_numeric(raw_text, errors="coerce")
    malformed_mask = (years >= 10000) & (years <= 99999)
    if malformed_mask.any():
        lower, upper = 1600.0, 2030.0
        shortened = (years[malformed_mask] // 10).astype("Int64")
        plausible_shortened = shortened.between(lower, upper)
        years.loc[malformed_mask[malformed_mask].index[plausible_shortened]] = shortened[plausible_shortened].astype(float)
    return years


# ---------------------------------------------------------------------------
# Collapsed parcel-year state (shared by the hedonic and devchoice preps).
# ---------------------------------------------------------------------------
#
# Both model inputs are built at the collapsed parcel grain: one row per
# (``geo_pid``, ``fy``), so a condo building enters each dataset once instead
# of as one row per unit account.

NO_DEVELOPMENT = "No Development"
CONDO_UNIT_LU = {"CD", "CP", "CC"}
RESIDENTIAL_LU = {"R1", "R2", "R3", "R4", "A", "CD", "CP", "CC", "CM", "RC", "RL"}

NUMERIC_COLS = ["living_area", "gross_area", "assessed_total", "assessed_land",
                "assessed_bldg", "land_sf", "yr_built", "res_units"]
# Point/time state variables carried from the representative row of the parcel-year.
STATE_COLS = ["zip_code", "overall_cond", "centroid_x", "centroid_y",
              "neighborhood_name", "median_hh_income", "emp_dist_m", "walkability",
              "zoning_use", "max_far", "max_height", "max_floors"]

# Same category structure as the single-year prep_for_devchoice.py.
PRIOR_LU_UNIT_CATEGORIES = {
    "R1": "1 Unit",
    "R2": "2 Units",
    "R3": "3 Units",
    "R4": "4-6 Units",
}

# Representative-row attributes the hedonic model needs beyond STATE_COLS.
# Included in the shared cached state so both preps read one cache file.
HEDONIC_REP_COLS = [
    "land_use_desc", "bldg_type", "structure_class",
    "bedrooms", "full_baths", "half_baths", "total_rooms",
    "num_parking", "num_floors",
]


def normalize_land_use(series: pd.Series) -> pd.Series:
    """Normalize assessor land-use codes, including compound source values."""
    return (
        series.astype("string")
        .str.strip()
        .str.upper()
        .str.split(r"\s*-\s*", n=1, expand=False)
        .str[0]
        .str.strip()
    )


# Two assessor schema generations: old FY16-20 use bare letters (A, G, E, ...),
# new FY21-25 use long codes ("A - Average", "EX - Excellent", ...). Map both
# to a single canonical letter so the condition categories are shared across
# eras (otherwise one-hot levels are disjoint by era and collinear).
CONDITION_MAP = {
    "EX": "E",   # Excellent
    "E": "E",
    "VG": "VG",  # Very Good
    "G": "G",    # Good
    "AVG": "A",  # Default - Average (new-gen junk)
    "A": "A",    # Average
    "F": "F",    # Fair
    "P": "P",    # Poor
    "VP": "VP",  # Very Poor
    "US": "US",  # Unsound
    "U": "US",
}


def normalize_condition(series: pd.Series) -> pd.Series:
    """Normalize overall-condition codes to a canonical letter shared across
    the two assessor schema generations (old bare letters vs new long codes)."""
    code = (
        series.astype("string")
        .str.strip()
        .str.upper()
        .str.split(r"\s*-\s*", n=1, expand=False)
        .str[0]
        .str.strip()
    )
    return code.map(CONDITION_MAP)


def categorize_units(units: pd.Series) -> pd.Series:
    """Return development-choice unit-count categories (NaN stays NaN)."""
    units = pd.to_numeric(units, errors="coerce")
    categories = pd.Series("21+ Units", index=units.index, dtype="string")
    categories.loc[units.le(0)] = NO_DEVELOPMENT
    categories.loc[units.eq(1)] = "1 Unit"
    categories.loc[units.eq(2)] = "2 Units"
    categories.loc[units.eq(3)] = "3 Units"
    categories.loc[units.between(4, 6)] = "4-6 Units"
    categories.loc[units.between(7, 20)] = "7-20 Units"
    categories.loc[units.isna()] = pd.NA
    return categories


def parcel_year_state(panel: pd.DataFrame,
                      extra_rep_cols: list[str] | None = None) -> pd.DataFrame:
    """Collapse assessor rows to one state per (geo_pid, fy).

    Mirrors clean_parcels.py condo handling: CM row supplies attributes and
    unit count; CD/CP/CC unit rows are aggregated (sums) into the parcel.

    ``extra_rep_cols`` names additional columns to carry from the
    representative row (e.g. hedonic building attributes); columns already in
    ``STATE_COLS`` or missing from the panel are skipped.

    Returns a frame indexed by (``geo_pid``, ``fy``) with the representative
    row's ``lu`` (normalized land-use code), ``res_units``, ``parcel_id`` and
    state variables, the aggregated area/value sums, ``n_unit_rows``, the
    derived ``units`` count, and ``unit_category``.
    """
    df = panel.dropna(subset=["geo_pid"]).copy()
    df["lu"] = normalize_land_use(df["land_use_code"])
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["_is_unit"] = df["lu"].isin(CONDO_UNIT_LU)

    state_vars = [c for c in STATE_COLS if c in df.columns]
    if extra_rep_cols:
        state_vars += [c for c in extra_rep_cols
                       if c in df.columns and c not in state_vars]

    # Representative row per parcel-year: CM row first, then other non-unit
    # rows, then unit rows (pure-condo parcels).
    rep = (
        df.assign(
            _rank=df["_is_unit"].astype(int) * 2 - df["lu"].eq("CM").astype(int)
        )
        .sort_values(["geo_pid", "fy", "_rank"], kind="stable")
        .groupby(["geo_pid", "fy"], sort=True)
        .head(1)
        .set_index(["geo_pid", "fy"])
    )

    def _sum(s: pd.Series) -> float:
        return s.sum(min_count=1)

    agg = df.groupby(["geo_pid", "fy"], sort=True).agg(
        living_area=("living_area", _sum),
        gross_area=("gross_area", _sum),
        assessed_total=("assessed_total", _sum),
        assessed_land=("assessed_land", _sum),
        assessed_bldg=("assessed_bldg", _sum),
        land_sf=("land_sf", _sum),
        yr_built=("yr_built", "max"),
        n_unit_rows=("_is_unit", "sum"),
    )

    state = rep[["lu", "res_units", "parcel_id", *state_vars]].join(agg)

    # Pure condo-unit parcels (representative row is a CD/CP/CC unit account)
    # behave as a condo building whose unit count is the number of accounts.
    only_units = state["lu"].isin(CONDO_UNIT_LU)
    state.loc[only_units, "lu"] = "CM"

    # Unit count. res_units is only reliable on the CM (condo master) account;
    # on R1-R4 it is present but ~0 (assessor quirk), so the LU code is the
    # source of truth there. Precedence:
    #   RL      -> 0 (land)
    #   RC / A  -> area estimate ceil(living_area / 850) (never recorded)
    #   R1-R4   -> the LU code's implied count (1/2/3); R4 stays NaN (4-6 is a
    #              range, not a count) and falls through to the area estimate
    #   CM      -> recorded res_units, else the number of unit accounts
    #   other   -> recorded res_units when present, else the area estimate
    area_est = np.ceil(state["living_area"] / 850.0)

    units = pd.Series(np.nan, index=state.index, dtype="float64")
    units = units.mask(state["lu"].eq("RL"), 0.0)
    units = units.mask(state["lu"].isin({"RC", "A"}), area_est)
    units = units.mask(state["lu"].isin({"R1", "R2", "R3"}),
                       state["lu"].map({"R1": 1.0, "R2": 2.0, "R3": 3.0}))
    units = units.mask(state["lu"].eq("CM"), state["res_units"])
    units = units.fillna(state["n_unit_rows"].where(state["lu"].eq("CM")))
    # Any remaining gaps (R4, or CM/other with no recorded units) -> area.
    units = units.fillna(state["res_units"].where(pd.to_numeric(state["res_units"], errors="coerce") > 0))
    units = units.fillna(area_est)
    state["units"] = units

    state["unit_category"] = state["lu"].map(PRIOR_LU_UNIT_CATEGORIES)
    state["unit_category"] = state["unit_category"].fillna(
        categorize_units(state["units"])
    )
    return state.drop(columns=["_rank"], errors="ignore")


def load_or_build_parcel_state(panel_path: Path,
                               state_path: Path | None = None,
                               rebuild: bool = False,
                               extra_rep_cols: list[str] | None = None) -> pd.DataFrame:
    """Load the cached collapsed parcel-year state, building it if absent.

    The collapse runs once and is cached to ``state_path`` so both model-input
    preps (hedonic and devchoice) share a single pass over the assessor panel.
    Pass ``rebuild=True`` (or delete the cache) after the panel or enrichment
    steps change. ``extra_rep_cols`` is forwarded to :func:`parcel_year_state`
    and baked into the cache, so callers needing different extra columns should
    use distinct cache files.
    """
    panel_path = Path(panel_path)
    state_path = Path(state_path) if state_path is not None else None
    if state_path is not None and state_path.exists() and not rebuild:
        print(f"Reading cached parcel-year state: {state_path}")
        state = pd.read_parquet(state_path)
        return state.set_index(["geo_pid", "fy"]).sort_index()

    panel = pd.read_parquet(require_existing_path(panel_path, "Panel parquet"))
    print("Collapsing assessor rows to parcel-year state...")
    state = parcel_year_state(panel, extra_rep_cols=extra_rep_cols)
    if state_path is not None:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state.reset_index().to_parquet(state_path, index=False)
        print(f"  cached -> {state_path}")
    return state
