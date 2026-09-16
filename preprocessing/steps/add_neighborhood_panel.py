"""Add neighborhood tags to the assessor panel via spatial join with Boston boundaries.

Boundaries are static, so parcels are spatially joined once per unique
``geo_pid`` (using the parcel centroid from ``stack_assessors.py``) and the
result is broadcast to all fiscal-year rows of that parcel.

Usage
-----
    python preprocessing/steps/add_neighborhood_panel.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import geopandas as gpd
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from preprocessing.utils import require_existing_path

NEIGHBORHOOD_NAME_CANDIDATES = ["name", "neighborhood", "neighborhood_name", "NBHD", "NBH_NAME"]
NEIGHBORHOOD_ID_CANDIDATES = ["neighborhood_id", "id", "OBJECTID"]

PREP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = PREP_DIR.parent / "processed_data" / "assessor_panel.parquet"
DEFAULT_BOUNDARIES = PREP_DIR / "raw_data" / "boston_neighborhood_boundaries.geojson"
CENTROID_CRS = "EPSG:26986"  # matches stack_assessors.py


def choose_column(columns: list[str], candidates: list[str]) -> str | None:
    by_lower = {str(col).lower(): str(col) for col in columns}
    for candidate in candidates:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL,
                        help="Assessor panel parquet (read and updated in place).")
    parser.add_argument("--neighborhood-geojson", type=Path, default=DEFAULT_BOUNDARIES,
                        help="Neighborhood boundary polygons.")
    parser.add_argument("--neighborhood-column", type=str, default="neighborhood_name")
    parser.add_argument("--neighborhood-id-column", type=str, default="neighborhood_id")
    args = parser.parse_args()
    args.panel = require_existing_path(args.panel, "Panel parquet")
    args.neighborhood_geojson = require_existing_path(
        args.neighborhood_geojson, "Neighborhood boundary file")
    return args


def main() -> None:
    args = parse_args()

    print(f"Reading panel: {args.panel}")
    panel = pd.read_parquet(args.panel)

    print(f"Reading neighborhood boundaries: {args.neighborhood_geojson}")
    neighborhoods = gpd.read_file(args.neighborhood_geojson, engine="pyogrio")
    if neighborhoods.empty or neighborhoods.geometry.isna().all():
        raise ValueError("Neighborhood boundary file has no valid geometries.")

    name_col = choose_column(list(neighborhoods.columns), NEIGHBORHOOD_NAME_CANDIDATES)
    if name_col is None:
        raise ValueError(
            "Could not find a neighborhood name column. "
            f"Expected one of {NEIGHBORHOOD_NAME_CANDIDATES}."
        )
    id_col = choose_column(list(neighborhoods.columns), NEIGHBORHOOD_ID_CANDIDATES)

    # Unique parcels with a valid centroid — join once, broadcast to all years.
    parcels = (
        panel.loc[panel["centroid_x"].notna() & panel["centroid_y"].notna(), "geo_pid"]
        .dropna()
        .drop_duplicates()
        .to_frame()
    )
    centroids = panel.dropna(subset=["geo_pid", "centroid_x", "centroid_y"])
    centroids = centroids.groupby("geo_pid")[["centroid_x", "centroid_y"]].first()
    parcels = parcels.join(centroids, on="geo_pid")

    points = gpd.GeoDataFrame(
        parcels,
        geometry=gpd.points_from_xy(parcels["centroid_x"], parcels["centroid_y"]),
        crs=CENTROID_CRS,
    )
    if points.crs != neighborhoods.crs:
        neighborhoods = neighborhoods.to_crs(points.crs)

    join_cols = ["geometry", name_col] + ([id_col] if id_col else [])
    print(f"Joining {len(points):,} unique parcels to neighborhoods...")
    joined = gpd.sjoin(
        points[["geometry"]], neighborhoods[join_cols], how="left", predicate="within",
    )
    name_by_parcel = joined.groupby(joined.index)[name_col].first()
    parcels[args.neighborhood_column] = name_by_parcel.reindex(parcels.index)
    if id_col:
        parcels[args.neighborhood_id_column] = (
            joined.groupby(joined.index)[id_col].first().reindex(parcels.index)
        )

    # Broadcast back to the full panel on geo_pid.
    merge_cols = ["geo_pid", args.neighborhood_column] + (
        [args.neighborhood_id_column] if id_col else []
    )
    panel = panel.drop(columns=merge_cols[1:], errors="ignore").merge(
        parcels[merge_cols], on="geo_pid", how="left",
    )

    panel.to_parquet(args.panel, index=False)
    matched = int(panel[args.neighborhood_column].notna().sum())
    print(f"Rows written: {len(panel):,}")
    print(f"Rows with neighborhood: {matched:,} ({matched / len(panel):.1%})")
    print(f"Output: {args.panel}")


if __name__ == "__main__":
    main()
