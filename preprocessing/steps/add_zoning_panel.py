"""Add zoning subdistrict attributes to the assessor panel via spatial join.

Zoning is static (one subdistrict shapefile), so parcels are joined once per
unique ``geo_pid`` (using the parcel centroid from ``stack_assessors.py``) and
the result is broadcast to all fiscal-year rows of that parcel.

Adds: ``zoning_use``, ``max_far``, ``max_height``, ``max_floors``.

Usage
-----
    python preprocessing/steps/add_zoning_panel.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import geopandas as gpd
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from preprocessing.utils import clean_numeric, require_existing_path

PREP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = PREP_DIR.parent / "processed_data" / "assessor_panel.parquet"
DEFAULT_ZONING = (
    PREP_DIR / "raw_data" / "boston_zoning_subdistricts" / "Boston_Zoning_Subdistricts.shp"
)
CENTROID_CRS = "EPSG:26986"  # matches stack_assessors.py

ZONING_COLUMN_MAP: dict[str, str] = {
    "Zoning_Sub": "zoning_use",
    "Max_FAR": "max_far",
    "Max_Height": "max_height",
    "Max_Number": "max_floors",
}
NUMERIC_ZONING = ["max_far", "max_height", "max_floors"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL,
                        help="Assessor panel parquet (read and updated in place).")
    parser.add_argument("--zoning-shapefile", type=Path, default=DEFAULT_ZONING,
                        help="Zoning subdistrict polygons.")
    args = parser.parse_args()
    args.panel = require_existing_path(args.panel, "Panel parquet")
    args.zoning_shapefile = require_existing_path(args.zoning_shapefile, "Zoning shapefile")
    return args


def main() -> None:
    args = parse_args()

    print(f"Reading panel: {args.panel}")
    panel = pd.read_parquet(args.panel)

    print(f"Reading zoning subdistricts: {args.zoning_shapefile}")
    zoning = gpd.read_file(
        args.zoning_shapefile, columns=[*ZONING_COLUMN_MAP.keys(), "geometry"],
    )
    zoning = zoning.rename(columns=ZONING_COLUMN_MAP)
    for col in NUMERIC_ZONING:
        zoning[col] = clean_numeric(zoning[col])

    # Unique parcels with a valid centroid — join once, broadcast to all years.
    parcels = panel.dropna(subset=["geo_pid", "centroid_x", "centroid_y"])
    parcels = parcels.groupby("geo_pid")[["centroid_x", "centroid_y"]].first()
    points = gpd.GeoDataFrame(
        parcels,
        geometry=gpd.points_from_xy(parcels["centroid_x"], parcels["centroid_y"]),
        crs=CENTROID_CRS,
    )
    if points.crs != zoning.crs:
        zoning = zoning.to_crs(points.crs)

    print(f"Joining {len(points):,} unique parcels to {len(zoning):,} zoning polygons...")
    joined = gpd.sjoin(
        points[["geometry"]], zoning[[*ZONING_COLUMN_MAP.values(), "geometry"]],
        how="left", predicate="within",
    )
    joined = joined.drop(columns=["index_right"], errors="ignore")
    joined = joined[~joined.index.duplicated(keep="first")]

    matched = joined["zoning_use"].notna().sum()
    print(f"  matched {matched:,} of {len(joined):,} parcels "
          f"({matched / len(joined):.1%})")

    panel = panel.drop(columns=[*ZONING_COLUMN_MAP.values()], errors="ignore").merge(
        joined.drop(columns="geometry").reset_index(), on="geo_pid", how="left",
    )

    panel.to_parquet(args.panel, index=False)
    print(f"Panel updated in place: {args.panel}")


if __name__ == "__main__":
    main()
