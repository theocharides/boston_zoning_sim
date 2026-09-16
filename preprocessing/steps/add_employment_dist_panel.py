"""Add employment-center distance to the assessor panel.

Straight-line distance in meters from each parcel centroid to the nearest
major employment center/CBD. Centers are static, so the distance is computed
once per unique ``geo_pid`` and broadcast to all fiscal-year rows.

Usage
-----
    python preprocessing/steps/add_employment_dist_panel.py
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

from preprocessing.utils import require_existing_path

PREP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = PREP_DIR.parent / "processed_data" / "assessor_panel.parquet"
CENTROID_CRS = "EPSG:26986"

DEFAULT_CENTERS = [
    {"name": "Downtown Boston CBD", "lat": 42.3555, "lon": -71.0605},
    {"name": "Back Bay", "lat": 42.3493, "lon": -71.0799},
    {"name": "Seaport", "lat": 42.3503, "lon": -71.0446},
    {"name": "Longwood Medical Area", "lat": 42.3360, "lon": -71.1057},
    {"name": "Kendall Square", "lat": 42.3626, "lon": -71.0863},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL,
                        help="Assessor panel parquet (read and updated in place).")
    parser.add_argument("--output-column", type=str, default="emp_dist_m")
    args = parser.parse_args()
    args.panel = require_existing_path(args.panel, "Panel parquet")
    return args


def main() -> None:
    args = parse_args()

    print(f"Reading panel: {args.panel}")
    panel = pd.read_parquet(args.panel)

    # One distance per unique parcel — centroids already in a metric CRS.
    parcels = panel.dropna(subset=["geo_pid", "centroid_x", "centroid_y"])
    parcels = parcels.groupby("geo_pid")[["centroid_x", "centroid_y"]].first()

    centers = gpd.GeoDataFrame(
        pd.DataFrame(DEFAULT_CENTERS),
        geometry=gpd.points_from_xy(
            [c["lon"] for c in DEFAULT_CENTERS],
            [c["lat"] for c in DEFAULT_CENTERS],
        ),
        crs="EPSG:4326",
    ).to_crs(CENTROID_CRS)

    xs = parcels["centroid_x"].to_numpy()
    ys = parcels["centroid_y"].to_numpy()
    nearest = np.full(len(parcels), np.inf)
    for center in centers.geometry:
        d = np.hypot(xs - center.x, ys - center.y)
        nearest = np.minimum(nearest, d)

    dist_by_parcel = pd.Series(nearest, index=parcels.index, name=args.output_column)

    panel = panel.drop(columns=[args.output_column], errors="ignore").merge(
        dist_by_parcel.reset_index(), on="geo_pid", how="left",
    )

    panel.to_parquet(args.panel, index=False)
    matched = int(panel[args.output_column].notna().sum())
    print(f"Centers used: {len(centers):,}")
    print(f"Rows written: {len(panel):,}")
    print(f"Rows with distance: {matched:,} ({matched / len(panel):.1%})")
    print(f"Output: {args.panel}")


if __name__ == "__main__":
    main()
