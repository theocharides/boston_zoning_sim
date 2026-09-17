"""Add a neighborhood walkability score to the assessor panel.

Panel-track port of the legacy ``neighborhood_walkability.py``: computes OSM
walking-network distance from each parcel to the nearest amenity in several
destination categories, converts distances to 0-100 category scores, and
averages them into one parcel ``walkability`` score.

Amenities are treated as static, so the score is computed once per unique
``geo_pid`` (parcel centroid from ``stack_assessors.py``) and broadcast to all
fiscal-year rows. The per-parcel scores are also cached to
``processed_data/walkability.parquet`` so the OSM download does not need to be
repeated when the panel is rebuilt.

Workflow:
1. Read unique parcel centroids from the assessor panel.
2. Download the walking network for the parcel extent from OpenStreetMap.
3. Download destination amenities from OpenStreetMap.
4. Snap parcels and destinations to the network.
5. Compute shortest-path network distance to the nearest destination per
   category.
6. Convert distances to category scores (linear decay from 100 at 0 m to 0 at
   ``--max-walk-distance-m``) and average them.

Usage
-----
    python preprocessing/steps/add_walkability_panel.py
    python preprocessing/steps/add_walkability_panel.py --sample-size 2000  # quick test
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from preprocessing.utils import require_existing_path

PREP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = PREP_DIR.parent / "processed_data" / "assessor_panel.parquet"
DEFAULT_CACHE = PREP_DIR.parent / "processed_data" / "walkability.parquet"
CENTROID_CRS = "EPSG:26986"  # matches stack_assessors.py

AMENITY_TAGS: dict[str, dict[str, object]] = {
    "grocery": {"shop": ["supermarket", "grocery", "convenience"]},
    "food": {"amenity": ["restaurant", "cafe", "fast_food"]},
    "education": {"amenity": ["school", "college", "university", "library"]},
    "park": {"leisure": ["park", "playground"]},
    "transit": {"public_transport": True, "railway": ["station", "tram_stop"],
                "highway": "bus_stop"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL,
                        help="Assessor panel parquet (read and updated in place).")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE,
                        help="Per-parcel walkability cache parquet.")
    parser.add_argument("--score-column", type=str, default="walkability",
                        help="Name of the output walkability score column.")
    parser.add_argument("--max-walk-distance-m", type=float, default=1600.0,
                        help="Distance where a category score decays to zero.")
    parser.add_argument("--distance-decay-exponent", type=float, default=1.0,
                        help="Power on the normalized distance decay curve "
                             "(1.0 = linear).")
    parser.add_argument("--sample-size", type=int, default=0,
                        help="Score only this many parcels (fast validation run).")
    args = parser.parse_args()
    args.panel = require_existing_path(args.panel, "Panel parquet")
    args.cache = args.cache.expanduser().resolve()
    if args.distance_decay_exponent <= 0:
        raise ValueError("distance_decay_exponent must be > 0")
    return args


def _linear_distance_score(distance_m: pd.Series, max_distance_m: float,
                           decay_exponent: float) -> pd.Series:
    """Convert distances to 0-100 scores with configurable power decay."""
    normalized = (distance_m / max_distance_m).clip(lower=0.0)
    score = 100.0 * (1.0 - np.power(normalized, decay_exponent))
    return score.clip(lower=0.0, upper=100.0).where(distance_m.notna())


def _flatten_feature_columns(features: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Reset multi-index columns returned by OSMnx feature queries."""
    out = features.copy()
    out.columns = [
        "_".join(str(part) for part in col if str(part) != "")
        if isinstance(col, tuple) else str(col)
        for col in out.columns
    ]
    return out


def _download_amenity_points(parcel_polygon) -> gpd.GeoDataFrame:
    """Download amenity features by category within the parcel extent."""
    amenity_frames: list[gpd.GeoDataFrame] = []
    for category, tags in AMENITY_TAGS.items():
        features = ox.features_from_polygon(parcel_polygon, tags=tags)
        if features.empty:
            continue
        features = _flatten_feature_columns(features)
        features = features[features.geometry.notna()].copy()
        features["geometry"] = features.geometry.map(
            lambda g: g if g.geom_type == "Point" else g.representative_point()
        )
        if features.empty:
            continue
        features = features[["geometry"]].copy()
        features["category"] = category
        amenity_frames.append(features)

    if not amenity_frames:
        raise ValueError("No walkability destination features downloaded from OpenStreetMap.")
    return pd.concat(amenity_frames, ignore_index=True)


def compute_walkability(parcel_points: gpd.GeoDataFrame, score_column: str,
                        max_walk_distance_m: float,
                        decay_exponent: float) -> pd.Series:
    """Score parcels (point GeoDataFrame, EPSG:26986) by network proximity to amenities."""
    hull = parcel_points.to_crs("EPSG:4326").unary_union.convex_hull

    print("Downloading walking network from OpenStreetMap...")
    graph = ox.graph_from_polygon(hull.buffer(0.01), network_type="walk", simplify=True)
    if len(graph.nodes) == 0:
        raise ValueError("Downloaded walking network has no nodes.")
    graph = ox.project_graph(graph)
    graph_crs = graph.graph.get("crs")
    parcel_points = parcel_points.to_crs(graph_crs)

    print("Downloading walk-destination amenities from OpenStreetMap...")
    amenities = _download_amenity_points(hull.buffer(0.01))
    amenities = gpd.GeoDataFrame(amenities, geometry="geometry", crs="EPSG:4326").to_crs(graph_crs)

    parcel_node_ids = ox.distance.nearest_nodes(
        graph,
        X=parcel_points.geometry.x.to_numpy(),
        Y=parcel_points.geometry.y.to_numpy(),
    )
    parcel_node_series = pd.Series(parcel_node_ids, index=parcel_points.index)

    category_scores: list[pd.Series] = []
    print("Computing network distance to nearest amenities by category...")
    for category in sorted(amenities["category"].unique()):
        amenity_subset = amenities.loc[amenities["category"] == category]
        amenity_node_ids = ox.distance.nearest_nodes(
            graph,
            X=amenity_subset.geometry.x.to_numpy(),
            Y=amenity_subset.geometry.y.to_numpy(),
        )
        amenity_node_ids = pd.Index(pd.Series(amenity_node_ids).dropna().astype(int).unique())
        if amenity_node_ids.empty:
            continue

        distances_to_targets = nx.multi_source_dijkstra_path_length(
            graph, sources=list(amenity_node_ids), weight="length",
        )
        parcel_distances = parcel_node_series.map(distances_to_targets)
        category_score = _linear_distance_score(
            parcel_distances.astype(float), max_walk_distance_m, decay_exponent,
        )
        category_score.name = f"walk_{category}_score"
        category_scores.append(category_score)

    if not category_scores:
        raise ValueError("No walkability category scores could be computed.")

    score_components = pd.concat(category_scores, axis=1)
    final_score = score_components.mean(axis=1, skipna=True)
    final_score.name = score_column
    return final_score.round(2)


def main() -> None:
    args = parse_args()

    print(f"Reading panel: {args.panel}")
    panel = pd.read_parquet(args.panel)

    # One score per unique parcel — centroids already in a metric CRS.
    parcels = panel.dropna(subset=["geo_pid", "centroid_x", "centroid_y"])
    parcels = parcels.groupby("geo_pid")[["centroid_x", "centroid_y"]].first()
    if args.sample_size and 0 < args.sample_size < len(parcels):
        parcels = parcels.sample(n=args.sample_size, random_state=42)

    parcel_points = gpd.GeoDataFrame(
        parcels,
        geometry=gpd.points_from_xy(parcels["centroid_x"], parcels["centroid_y"]),
        crs=CENTROID_CRS,
    )

    scores = compute_walkability(
        parcel_points, args.score_column,
        args.max_walk_distance_m, args.distance_decay_exponent,
    )

    # Cache per-parcel scores so a panel rebuild can skip the OSM download.
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    cache = scores.rename_axis("geo_pid").reset_index()
    if args.cache.exists():
        existing = pd.read_parquet(args.cache)
        existing = existing[~existing["geo_pid"].isin(cache["geo_pid"])]
        cache = pd.concat([existing, cache], ignore_index=True)
    cache.to_parquet(args.cache, index=False)
    print(f"Per-parcel scores cached to {args.cache}")

    panel = panel.drop(columns=[args.score_column], errors="ignore").merge(
        cache, on="geo_pid", how="left",
    )
    panel.to_parquet(args.panel, index=False)

    n_scored = int(panel[args.score_column].notna().sum())
    print(f"Panel rows with walkability score: {n_scored:,} of {len(panel):,}")
    print(f"Panel updated in place: {args.panel}")


if __name__ == "__main__":
    main()
