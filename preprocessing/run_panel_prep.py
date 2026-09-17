"""Orchestrate the multi-year preprocessing track.

Runs, in order:
    1. stack_assessors.py           – build the FY2016–FY2025 panel + geometry
    2. add_neighborhood_panel.py    – neighborhood tags (spatial join)
    3. add_income_panel.py          – ACS tract median household income
    4. add_employment_dist_panel.py – distance to nearest employment center
    5. add_zoning_panel.py          – zoning subdistrict attributes
    6. add_walkability_panel.py     – OSM network walkability score
    7. prep_for_hedonic_panel.py    – hedonic input: parcel-year panel with the
                                      300 m spatio-temporal precedence variable
    8. prep_for_devchoice_spatial.py – dev-choice input: 5 pooled 1-year
                                       cross-sections with 600 m spatially
                                       lagged development activity

Outputs (two separate datasets, per the project README):
    processed_data/hedonic_input.parquet   – parcel panel for the hedonic model
    processed_data/devchoice_input.parquet – cross-sectional developer choices

Usage
-----
    python preprocessing/run_panel_prep.py
    python preprocessing/run_panel_prep.py --skip-income        # no Census key
    python preprocessing/run_panel_prep.py --skip-walkability   # no OSM download
    python preprocessing/run_panel_prep.py --years 2024 2025    # subset years
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

STEPS_DIR = Path(__file__).resolve().parent / "steps"


def run_step(name: str, script: Path, extra_args: list[str] | None = None) -> None:
    cmd = [sys.executable, str(script), *(extra_args or [])]
    print(f"\n{'=' * 70}\nRunning: {name}\n{'=' * 70}")
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--years", type=int, nargs="*", default=None,
                        help="Fiscal years for the stacking step (default: all).")
    parser.add_argument("--skip-income", action="store_true",
                        help="Skip the ACS income step (no Census API key needed).")
    parser.add_argument("--census-api-key", type=str, default=None,
                        help="Forwarded to add_income_panel.py.")
    parser.add_argument("--skip-walkability", action="store_true",
                        help="Skip the OSM walkability step (large download).")
    parser.add_argument("--skip-prep", action="store_true",
                        help="Only build/enrich the panel; skip hedonic prep.")
    parser.add_argument("--rebuild-state", action="store_true",
                        help="Rebuild the cached collapsed parcel-year state "
                             "(processed_data/parcel_state.parquet) instead of "
                             "reusing it. Use after re-stacking or enrichment.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    stack_args = ["--years", *[str(y) for y in args.years]] if args.years else None
    run_step("stack assessors", STEPS_DIR / "stack_assessors.py", stack_args)
    run_step("add neighborhood", STEPS_DIR / "add_neighborhood_panel.py")

    if args.skip_income:
        print("\nSkipping income step (no Census API key).")
    else:
        income_args = (
            ["--census-api-key", args.census_api_key] if args.census_api_key else None
        )
        run_step("add income", STEPS_DIR / "add_income_panel.py", income_args)

    run_step("add employment distance", STEPS_DIR / "add_employment_dist_panel.py")
    run_step("add zoning", STEPS_DIR / "add_zoning_panel.py")

    if args.skip_walkability:
        print("\nSkipping walkability step (OSM download disabled).")
    else:
        run_step("add walkability", STEPS_DIR / "add_walkability_panel.py")

    if not args.skip_prep:
        # Two separate end products (see README):
        #   hedonic_input.parquet   – the parcel-year panel (PID-anchored)
        #   devchoice_input.parquet – 5 pooled 1-year cross-sections
        #                             (spatially anchored via polygon overlay)
        # Both read the shared collapsed parcel-year state cached at
        # processed_data/parcel_state.parquet. The hedonic prep runs first and
        # builds the cache (with the hedonic representative-row attributes);
        # the devchoice prep reuses it. Rebuild the cache whenever the panel or
        # enrichment steps change (--rebuild-state).
        state_args = ["--rebuild"] if args.rebuild_state else None
        run_step("prep for hedonic", STEPS_DIR / "prep_for_hedonic_panel.py", state_args)
        run_step("prep for devchoice", STEPS_DIR / "prep_for_devchoice_spatial.py", state_args)


if __name__ == "__main__":
    main()
