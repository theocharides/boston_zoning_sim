"""
Stack Boston property assessment files (FY2016-FY2025) into a parcel-year panel,
attach per-year parcel geometry, and add fiscal-year dummies.

Geometry
--------
Each fiscal year of assessor records is joined to that year's parcel polygons
(``Parcels__YYYY_.geojson``). Assessor account IDs (including condo units) are
normalized to 10-digit strings and mapped to polygon parcel IDs for the 
appropriate year.

Two outputs are written:

  - ``assessor_panel.parquet``: the stacked panel. Carries ``geo_pid`` (matched
    polygon parcel id) and parcel centroids (``centroid_x`` / ``centroid_y``,
    EPSG:26986 meters) so distance can be built for the spatio-temporal lag.
  - ``parcel_geometry/parcels_fyYYYY.parquet``: one row per parcel per year
    with the polygon (GeoParquet, EPSG:4326), used by the devchoice prep for
    the consecutive-year polygon overlay that maps each parcel to its
    prior-year predecessor. Read the whole folder at once with
    ``geopandas.read_parquet`` on the directory or via ``pyarrow.dataset``.`

Usage
-----
    python preprocessing/steps/stack_assessors.py
    python preprocessing/steps/stack_assessors.py --years 2024 2025
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from preprocessing.utils import (
    clean_numeric,
    normalize_condition,
    normalize_pid,
    require_existing_path,
    to_base_pid,
)

PREP_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PREP_DIR / "raw_data" / "assessors"
SHAPE_DIR = PREP_DIR / "raw_data" / "shapes"
OUT_DIR = PREP_DIR.parent / "processed_data"
OUT_PANEL = OUT_DIR / "assessor_panel.parquet"
OUT_GEOMETRY_DIR = OUT_DIR / "parcel_geometry"

CENTROID_CRS = "EPSG:26986"  # MA State Plane Mainland (meters) for distance work

# Parcel-id column in each year's shapefile (schema differs by vintage).
SHAPE_ID_COL = {
    **{fy: "PID_LONG" for fy in range(2016, 2020)},
    **{fy: "MAP_PAR_ID" for fy in range(2020, 2026)},
}

# ---------------------------------------------------------------------------
# 1. Harmonisation map: unified name -> {old (FY16-20) name, new (FY21-25) name}
# ---------------------------------------------------------------------------
COLUMN_MAP = {
    "parcel_id":        ("PID", "PID"),
    "gis_id":           ("GIS_ID", "GIS_ID"),
    "zip_code":         ("ZIPCODE", "ZIP_CODE"),
    "land_use_code":    ("LU", "LU"),
    "land_use_desc":    (None, "LU_DESC"),
    "bldg_type":        (None, "BLDG_TYPE"),
    "assessed_land":    ("AV_LAND", "LAND_VALUE"),
    "assessed_bldg":    ("AV_BLDG", "BLDG_VALUE"),
    "assessed_total":   ("AV_TOTAL", "TOTAL_VALUE"),
    "land_sf":          ("LAND_SF", "LAND_SF"),
    "gross_area":       ("GROSS_AREA", "GROSS_AREA"),
    "living_area":      ("LIVING_AREA", "LIVING_AREA"),
    "yr_built":         ("YR_BUILT", "YR_BUILT"),
    "yr_remodel":       ("YR_REMOD", "YR_REMODEL"),
    "structure_class":  ("STRUCTURE_CLASS", "STRUCTURE_CLASS"),
    "overall_cond":     ("R_OVRALL_CND", "OVERALL_COND"),
    "bedrooms":         ("R_BDRMS", "BED_RMS"),
    "full_baths":       ("R_FULL_BTH", "FULL_BTH"),
    "half_baths":       ("R_HALF_BTH", "HLF_BTH"),
    "total_rooms":      ("R_TOTAL_RMS", "TT_RMS"),
    "com_units":        ("S_UNIT_COM", "COM_UNITS"),
    "num_floors":       ("NUM_FLOORS", "RES_FLOOR"),
    "num_parking":      ("U_NUM_PARK", "NUM_PARKING"),
}

OLD_GEN_YEARS = range(2016, 2021)  # files with AV_* / R_* / S_* schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--years",
        type=int,
        nargs="*",
        default=None,
        help="Fiscal years to process (default: all found in raw_data/assessors).",
    )
    return parser.parse_args()


def load_year(path: Path, fy: int) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df.columns = df.columns.str.strip()  # ' GROSS_TAX ' -> 'GROSS_TAX'
    old_gen = fy in OLD_GEN_YEARS

    out = pd.DataFrame()
    for unified, (old_name, new_name) in COLUMN_MAP.items():
        src = old_name if old_gen else new_name
        if src and src in df.columns:
            out[unified] = df[src]
        elif src and src.strip() in df.columns:
            out[unified] = df[src.strip()]
        else:
            out[unified] = pd.NA

    out["parcel_id"] = normalize_pid(out["parcel_id"])
    # Normalize condition codes so the two schema generations share categories.
    out["overall_cond"] = normalize_condition(out["overall_cond"])
    out["fy"] = fy
    return out


def load_geometry(fy: int) -> gpd.GeoDataFrame:
    """Load one year of parcel polygons, normalized to a deduped parcel key."""
    path = SHAPE_DIR / f"Parcels__{fy}_.geojson"
    id_col = SHAPE_ID_COL[fy]

    gdf = gpd.read_file(path, columns=[id_col], engine="pyogrio")
    gdf = gdf.rename(columns={id_col: "geo_pid"})
    gdf["geo_pid"] = normalize_pid(gdf["geo_pid"])
    gdf = gdf.drop_duplicates(subset=["geo_pid"], keep="first")

    # Point-on-surface centroids in a projected CRS for distance-based weights.
    points = gdf.geometry.representative_point()
    gdf["centroid_x"] = points.to_crs(CENTROID_CRS).x
    gdf["centroid_y"] = points.to_crs(CENTROID_CRS).y
    return gdf


def attach_geometry(df: pd.DataFrame, geo: gpd.GeoDataFrame) -> pd.DataFrame:
    """Map assessor rows to parcel polygons (direct id, then base parcel)."""
    parcel_keys = set(geo["geo_pid"])
    base_pid = to_base_pid(df["parcel_id"])
    df["geo_pid"] = np.where(
        df["parcel_id"].isin(parcel_keys),
        df["parcel_id"],
        np.where(base_pid.isin(parcel_keys), base_pid, pd.NA),
    )
    return df.merge(
        geo[["geo_pid", "centroid_x", "centroid_y"]],
        on="geo_pid",
        how="left",
    )


def add_year_dummies(panel: pd.DataFrame) -> pd.DataFrame:
    """Add fiscal-year dummies (FY2016 is the base level).

    The assessor data has no quarterly transactions, so the time fixed
    effects are yearly, not quarterly."""
    panel = panel.sort_values(["parcel_id", "fy"]).reset_index(drop=True)
    yd = pd.get_dummies(panel["fy"].astype(int), prefix="y", dtype=int)
    base = f"y_{int(panel['fy'].min())}"
    return pd.concat([panel, yd.drop(columns=[base])], axis=1)


def main() -> None:
    args = parse_args()

    files = sorted(RAW_DIR.glob("property-assessment-fy*.csv"))
    if not files:
        raise FileNotFoundError(f"No assessor CSVs found in {RAW_DIR}")
    if args.years:
        files = [f for f in files if int(f.stem[-4:]) in args.years]

    OUT_GEOMETRY_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
    for f in files:
        fy = int(f.stem[-4:])

        df = load_year(f, fy)
        geo = load_geometry(fy)
        df = attach_geometry(df, geo)

        n_matched = df["geo_pid"].notna().sum()
        print(
            f"FY{fy}: {len(df):,} rows, {df['parcel_id'].nunique():,} accounts, "
            f"{n_matched:,} matched to {len(geo):,} polygons "
            f"({n_matched / len(df):.1%})"
        )
        frames.append(df)

        # Per-year polygon store (GeoParquet part) for the devchoice overlay.
        geo_part = gpd.GeoDataFrame(
            geo[["geo_pid", "geometry"]].rename(columns={"geo_pid": "parcel_id"}),
            geometry="geometry",
            crs=geo.crs,
        )
        geo_part.insert(1, "fy", fy)
        geo_part.to_parquet(OUT_GEOMETRY_DIR / f"parcels_fy{fy}.parquet", index=False)
        del geo, geo_part

    panel = pd.concat(frames, ignore_index=True)

    # type cleaning
    for col in ["assessed_total", "assessed_land", "assessed_bldg", "living_area",
                "gross_area", "land_sf"]:
        panel[col] = clean_numeric(panel[col])

    panel = add_year_dummies(panel)

    OUT_PANEL.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT_PANEL, index=False)
    print(f"\nPanel written to {OUT_PANEL}")
    print(f"Geometry parts written to {OUT_GEOMETRY_DIR}")
    print(f"  rows={len(panel):,}  parcels={panel['parcel_id'].nunique():,}  "
          f"years={sorted(panel['fy'].unique())}")
    print(panel[["parcel_id", "fy", "geo_pid", "centroid_x", "centroid_y",
                 "assessed_total"]].head(10).to_string())


if __name__ == "__main__":
    main()
