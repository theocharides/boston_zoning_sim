"""Add tract median household income to the assessor panel.

Downloads ACS 5-year tract-level median household income (B19013_001E) and
joins it to parcel centroids spatially. The join is computed once per unique
``geo_pid`` per ACS vintage and broadcast to all matching panel rows.

ACS year mapping: assessor fiscal year FY is based on a January 1 valuation in
calendar year FY-1, so income is taken from the ACS 5-year vintage ending in
FY-2 (e.g. FY2025 -> ACS 2023 5-year). Rows whose fiscal year has no published
ACS vintage are left as NaN.

The resulting panel column is time-varying but smoothed: each parcel-year gets
its own vintage's value, yet consecutive vintages overlap ~80% (ACS 2019 spans
2015-2019, ACS 2020 spans 2016-2020).

Usage
-----
    python preprocessing/steps/add_income_panel.py --census-api-key KEY
    CENSUS_API_KEY=... python preprocessing/steps/add_income_panel.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import ssl
import sys
from json import JSONDecodeError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import geopandas as gpd
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from preprocessing.utils import require_existing_path

PREP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = PREP_DIR.parent / "processed_data" / "assessor_panel.parquet"
CENTROID_CRS = "EPSG:26986"

# Assessor FY -> ACS 5-year vintage to use (FY values are as of Jan 1 of FY-1;
# ACS 5-year estimates lag ~2 years behind that).
FY_TO_ACS = {fy: fy - 2 for fy in range(2016, 2026)}


def _https_context() -> ssl.SSLContext:
    """Build an HTTPS context that works even when the Windows cert store fails.

    Some conda Python builds on Windows raise ``ssl.SSLError: NOT_ENOUGH_DATA``
    when loading certificates from the Windows store. The failure surfaces when
    the context loads certs (not at creation), so prefer certifi's bundle when
    it is available rather than relying on a try/except around the failing call.
    """
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL,
                        help="Assessor panel parquet (read and updated in place).")
    parser.add_argument("--state-fips", type=str, default="25")
    parser.add_argument("--county-fips", type=str, default="025")
    parser.add_argument("--income-column", type=str, default="median_hh_income")
    parser.add_argument("--census-api-key", type=str, default=None,
                        help="Defaults to CENSUS_API_KEY env var.")
    args = parser.parse_args()
    args.panel = require_existing_path(args.panel, "Panel parquet")
    args.state_fips = args.state_fips.zfill(2)
    args.county_fips = args.county_fips.zfill(3)
    return args


def fetch_acs_income(acs_year: int, state_fips: str, county_fips: str,
                     census_api_key: str | None) -> pd.DataFrame:
    """Fetch tract median household income from the ACS 5-year API."""
    base_url = f"https://api.census.gov/data/{acs_year}/acs/acs5"
    query = {
        "get": "B19013_001E",
        "for": "tract:*",
        "in": f"state:{state_fips} county:{county_fips}",
    }
    if census_api_key:
        query["key"] = census_api_key

    url = f"{base_url}?{urlencode(query)}"
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, context=_https_context()) as response:
        body = response.read().decode("utf-8", errors="replace")

    try:
        payload = json.loads(body)
    except JSONDecodeError as exc:
        preview = body.strip().replace("\n", " ")[:240]
        raise ValueError(
            "ACS API returned a non-JSON response (missing/invalid Census API key?). "
            f"Response preview: {preview}"
        ) from exc

    if isinstance(payload, dict) and "error" in payload:
        raise ValueError(f"ACS API error: {payload['error']}")
    if not payload or len(payload) < 2:
        raise ValueError(f"ACS {acs_year}: no tract rows returned.")

    df = pd.DataFrame(payload[1:], columns=payload[0])
    df["GEOID"] = df["state"] + df["county"] + df["tract"]
    df["B19013_001E"] = pd.to_numeric(df["B19013_001E"], errors="coerce")
    return df[["GEOID", "B19013_001E"]]


def load_tract_geometry(acs_year: int, state_fips: str, county_fips: str) -> gpd.GeoDataFrame:
    """Download tract geometry from TIGER/Line and filter to the county."""
    import io
    import tempfile
    import zipfile

    tiger_url = (
        f"https://www2.census.gov/geo/tiger/TIGER{acs_year}/TRACT/"
        f"tl_{acs_year}_{state_fips}_tract.zip"
    )
    request = Request(tiger_url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, context=_https_context()) as response:
        zf = zipfile.ZipFile(io.BytesIO(response.read()))
    with tempfile.TemporaryDirectory() as tmpdir:
        zf.extractall(tmpdir)
        shp_path = next(Path(tmpdir).glob("*.shp"))
        tracts = gpd.read_file(shp_path, engine="pyogrio")
    if "COUNTYFP" not in tracts.columns or "GEOID" not in tracts.columns:
        raise ValueError("Unexpected tract geometry schema from TIGER source.")
    tracts = tracts[tracts["COUNTYFP"] == county_fips].copy()
    if tracts.empty:
        raise ValueError(f"No tract geometry rows for county FIPS {county_fips}.")
    return tracts[["GEOID", "geometry"]]


def main() -> None:
    args = parse_args()

    census_api_key = args.census_api_key or os.getenv("CENSUS_API_KEY")
    if not census_api_key:
        raise ValueError(
            "Census API key is required. Set CENSUS_API_KEY or pass --census-api-key."
        )

    print(f"Reading panel: {args.panel}")
    panel = pd.read_parquet(args.panel)

    # Unique (geo_pid, fy) combos that need income — join once, then broadcast.
    parcels = panel.dropna(subset=["geo_pid", "centroid_x", "centroid_y"])
    parcels = parcels[["geo_pid", "fy", "centroid_x", "centroid_y"]].drop_duplicates()
    parcels["acs_year"] = parcels["fy"].map(FY_TO_ACS)

    results: list[pd.DataFrame] = []
    for acs_year, group in parcels.groupby("acs_year"):
        print(f"Fetching ACS {acs_year} tract income...")
        income = fetch_acs_income(acs_year, args.state_fips, args.county_fips, census_api_key)
        tracts = load_tract_geometry(acs_year, args.state_fips, args.county_fips)
        tracts = tracts.merge(income, on="GEOID", how="left")

        points = gpd.GeoDataFrame(
            group,
            geometry=gpd.points_from_xy(group["centroid_x"], group["centroid_y"]),
            crs=CENTROID_CRS,
        )
        if points.crs != tracts.crs:
            tracts = tracts.to_crs(points.crs)

        print(f"  Joining {len(points):,} parcels to {len(tracts):,} tracts...")
        joined = gpd.sjoin(
            points[["geo_pid", "fy", "geometry"]],
            tracts[["B19013_001E", "geometry"]],
            how="left",
            predicate="within",
        )
        by_parcel = joined.groupby(["geo_pid", "fy"])["B19013_001E"].first()
        results.append(by_parcel.reset_index().rename(
            columns={"B19013_001E": args.income_column}))

    income_all = pd.concat(results, ignore_index=True)
    panel = panel.drop(columns=[args.income_column], errors="ignore").merge(
        income_all, on=["geo_pid", "fy"], how="left",
    )
    panel[args.income_column] = panel[args.income_column].astype("float64")

    panel.to_parquet(args.panel, index=False)
    matched = int(panel[args.income_column].notna().sum())
    print(f"Rows written: {len(panel):,}")
    print(f"Rows with income: {matched:,} ({matched / len(panel):.1%})")
    print(f"Output: {args.panel}")


if __name__ == "__main__":
    main()
