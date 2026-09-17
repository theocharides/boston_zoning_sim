"""Hedonic price-model workflow: fit OLS specs and report fit stats, VIF, and Moran's I.

Reads ``processed_data/hedonic_input.parquet`` (the collapsed residential
parcel-year panel) and estimates a log-linear hedonic model::

    log(price_per_sqft) ~ <variable set> + year dummies

The outcome and the yearly fixed effects (``y_2017``…``y_2025``) are always
included; ``SPECS`` below names the variable sets to compare. For each spec the
workflow reports AIC, BIC, and a VIF table (year dummies excluded — they are
controls, not part of the set under test), plus Moran's I of the residuals
over parcel centroids to flag leftover spatial autocorrelation. Each spec is
also evaluated out-of-sample with a K-fold spatial split: parcels are cut
into contiguous centroid bands *within each neighborhood*, so every
neighborhood appears in every fold and its dummies stay estimable.

Usage
-----
    python models/hedonic.py                      # all specs in SPECS
    python models/hedonic.py --spec baseline
    python models/hedonic.py --sample 100000      # fast iteration on a subset
    python models/hedonic.py --spatial-folds 0    # skip spatial CV

Outputs (coefficient table, fit stats, VIFs, spatial-CV metrics) are written
as per-spec CSVs (``hedonic_*_{spec}.csv``) to ``models/output/`` (override
with ``--out-dir``). Each spec's files are saved as soon as that spec
finishes, so an interrupted run keeps everything already completed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from preprocessing.utils import require_existing_path

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "processed_data"
DEFAULT_INPUT = PROCESSED_DIR / "hedonic_input.parquet"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "output"

OUTCOME = "price_per_sqft"
YEAR_DUMMIES = [f"y_{y}" for y in range(2017, 2026)]

# Moran's I on OLS residuals: kNN spatial weights over parcel centroids,
# permutation-based p-value (kNN weights are not symmetric).
MORAN_K = 8
MORAN_PERMUTATIONS = 499

# Continuous variables entered in logs (positive, right-skewed).
LOG_VARS = {"living_area", "gross_area", "land_sf", "median_hh_income",
            "emp_dist_m", "prior_sales_avg"}
# Continuous variables entered in levels.
LEVEL_VARS = {"yr_built", "num_floors", "units", "walkability"}
# Count-like fields stored as text in the assessor data -> coerced to numeric.
NUMERIC_TEXT_VARS = {"bedrooms", "full_baths", "half_baths", "total_rooms",
                     "num_parking"}
# Everything else named in a spec is treated as categorical (dummy-encoded,
# first level dropped) — land_use_code, bldg_type, overall_cond,
# structure_class, neighborhood_name, zip_code, ...

# Variable sets to compare. Year dummies are always added on top.
SPECS: dict[str, list[str]] = {
    "baseline": [
        "land_use_code",
        "prior_sales_avg",
        "emp_dist_m", 
        "walkability",
        "median_hh_income", 
    ],
    "neighborhood": [
        "land_use_code",
        "prior_sales_avg",
        "emp_dist_m", 
        "walkability",
        "median_hh_income", 
        "neighborhood_name"
    ],
    "building": [
        "land_use_code",
        "prior_sales_avg",
        "emp_dist_m", 
        "neighborhood_name",
        "living_area",

    ],
    "building2": [
        "land_use_code",
        "prior_sales_avg",
        "emp_dist_m", 
        "neighborhood_name",
        "living_area",
        "yr_built"
    ]
}


def build_design(df: pd.DataFrame, variables: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Assemble the design matrix for a variable set plus year dummies.

    Returns (X, y) with y = log(price_per_sqft). Rows with any missing or
    non-positive value in a log-transformed variable are dropped.
    """
    missing = [v for v in variables if v not in df.columns]
    if missing:
        raise KeyError(f"Variables not in hedonic input: {missing}")

    cols: dict[str, pd.Series] = {}
    for var in variables:
        s = df[var]
        if var in NUMERIC_TEXT_VARS:
            s = pd.to_numeric(s, errors="coerce")
            cols[var] = s
        elif var in LOG_VARS:
            s = pd.to_numeric(s, errors="coerce")
            cols[f"log_{var}"] = np.log(s.where(s > 0))
        elif var in LEVEL_VARS:
            cols[var] = pd.to_numeric(s, errors="coerce")
        else:  # categorical
            dummies = pd.get_dummies(s.astype("string"), prefix=var,
                                     dummy_na=True, drop_first=True, dtype=float)
            cols.update({c: dummies[c] for c in dummies.columns})

    X = pd.DataFrame(cols, index=df.index)
    for yd in YEAR_DUMMIES:
        X[yd] = df[yd].astype(float)

    y = np.log(pd.to_numeric(df[OUTCOME], errors="coerce").where(lambda s: s > 0))

    X = sm.add_constant(X, has_constant="add")
    data = X.join(y.rename("_outcome")).dropna()
    Xc = data.drop(columns="_outcome").astype(float)
    Xc = drop_degenerate_columns(Xc)
    return Xc, data["_outcome"].astype(float)


def drop_degenerate_columns(X: pd.DataFrame) -> pd.DataFrame:
    """Drop zero-variance and perfectly duplicate columns (singularity guards)."""
    keep_cols = ["const"] if "const" in X.columns else []
    body = X.drop(columns=["const"], errors="ignore")
    body = body.loc[:, body.std() > 0]               # all-zero / constant
    body = body.loc[:, ~body.T.duplicated()]          # exact duplicate columns
    return pd.concat([X[keep_cols], body], axis=1) if keep_cols else body


def compute_vif(X: pd.DataFrame) -> pd.Series:
    """VIF per regressor, excluding the constant and year dummies."""
    keep = [c for c in X.columns if c != "const" and c not in YEAR_DUMMIES]
    Xv = X[keep]
    # Drop zero-variance columns (VIF undefined).
    Xv = Xv.loc[:, Xv.std() > 0]
    # Keep a constant in the matrix so statsmodels' auxiliary regressions use
    # the centered R^2. Without one it uses the uncentered R^2, which inflates
    # VIFs for high-mean / low-variance columns (e.g. log income).
    Xv = sm.add_constant(Xv, has_constant="add")
    vif = {c: variance_inflation_factor(Xv.to_numpy(), i)
           for i, c in enumerate(Xv.columns) if c != "const"}
    return pd.Series(vif).sort_values(ascending=False)


def morans_i(residuals: pd.Series, df: pd.DataFrame, k: int = MORAN_K,
             permutations: int = MORAN_PERMUTATIONS, seed: int = 0) -> dict:
    """Moran's I of model residuals over parcel centroids (kNN weights).

    Residuals are averaged per ``geo_pid`` first: the panel repeats each
    parcel across years, and distance-0 duplicate coordinates would otherwise
    make a parcel its own neighbor. Inference is by permutation of the
    residuals (two-sided pseudo p-value).
    """
    nan_result = {"moran_i": np.nan, "moran_z": np.nan, "moran_p": np.nan,
                  "moran_n": 0}
    if not {"geo_pid", "centroid_x", "centroid_y"} <= set(df.columns):
        return nan_result
    d = df.loc[residuals.index, ["geo_pid", "centroid_x", "centroid_y"]].copy()
    d["resid"] = residuals
    d = d.dropna(subset=["centroid_x", "centroid_y"])
    g = d.groupby("geo_pid").agg(x=("centroid_x", "first"),
                                 y=("centroid_y", "first"),
                                 resid=("resid", "mean"))
    n = len(g)
    if n < k + 2:
        return nan_result

    coords = g[["x", "y"]].to_numpy()
    _, idx = cKDTree(coords).query(coords, k=k + 1)  # col 0 is self (dist 0)
    rows = np.repeat(np.arange(n), k)
    cols = idx[:, 1:].ravel()
    W = sparse.csr_matrix((np.full(n * k, 1.0 / k), (rows, cols)),
                          shape=(n, n))
    s0 = W.sum()

    z = g["resid"].to_numpy()
    z = z - z.mean()
    denom = z @ z
    if denom == 0:
        return nan_result

    def _stat(v: np.ndarray) -> float:
        return float((n / s0) * (v @ (W @ v)) / denom)

    obs = _stat(z)
    rng = np.random.default_rng(seed)
    sims = np.array([_stat(rng.permutation(z)) for _ in range(permutations)])
    sd = sims.std(ddof=1)
    z_sim = float((obs - sims.mean()) / sd) if sd > 0 else np.nan
    extreme = np.sum(np.abs(sims - sims.mean()) >= abs(obs - sims.mean()))
    p_sim = float((1 + extreme) / (1 + permutations))
    return {"moran_i": obs, "moran_z": z_sim, "moran_p": p_sim,
            "moran_n": n}


def assign_spatial_folds(df: pd.DataFrame, n_folds: int,
                         seed: int = 0) -> pd.Series:
    """Map each ``geo_pid`` to a spatial fold (0..n_folds-1).

    Within each neighborhood, parcels are sorted along the x or y centroid
    axis (alternating by neighborhood to dilute directional bias) and cut
    into ``n_folds`` contiguous bands. Because the cut is per neighborhood,
    every neighborhood with >= n_folds parcels appears in every fold, so
    categorical levels (e.g. neighborhood dummies) are always estimable on
    the training folds. Assignment is at parcel level, so a parcel's yearly
    rows never straddle train/test. Parcels lacking centroids are assigned
    at random (seeded).
    """
    parcels = df[["geo_pid", "neighborhood_name",
                  "centroid_x", "centroid_y"]].drop_duplicates("geo_pid")
    fold = pd.Series(np.nan, index=parcels["geo_pid"].to_numpy(), dtype=float)
    hoods = parcels["neighborhood_name"].astype("string").fillna("<none>")
    axes = ["centroid_x", "centroid_y"]
    for i, (_, sub) in enumerate(parcels.groupby(hoods)):
        have_xy = sub.dropna(subset=["centroid_x", "centroid_y"])
        order = have_xy.sort_values(axes[i % 2], kind="stable")
        bands = np.floor(np.arange(len(order)) * n_folds / max(len(order), 1))
        fold.loc[order["geo_pid"].to_numpy()] = bands.astype(int)
    missing = fold.isna()
    if missing.any():
        rng = np.random.default_rng(seed)
        fold.loc[fold.index[missing]] = rng.integers(0, n_folds,
                                                     int(missing.sum()))
    return fold.astype(int).rename("fold")


def spatial_cv(df: pd.DataFrame, name: str, variables: list[str],
               n_folds: int, seed: int = 0) -> pd.DataFrame:
    """Out-of-sample evaluation over the folds from assign_spatial_folds.

    Fits the spec on K-1 folds and predicts the held-out fold, rotating over
    folds; reports per-fold and pooled out-of-sample RMSE/MAE/R^2 (log-space
    outcome) plus Moran's I of the held-out residuals.
    """
    X, y = build_design(df, variables)
    fold_map = assign_spatial_folds(df.loc[X.index], n_folds, seed)
    row_fold = df.loc[X.index, "geo_pid"].map(fold_map).to_numpy()

    def _metrics(resid: pd.Series) -> dict:
        yv = y.loc[resid.index]
        sse = float(resid @ resid)
        sst = float(((yv - yv.mean()) ** 2).sum())
        return {"n_test": len(resid),
                "rmse": float(np.sqrt(np.mean(resid ** 2))),
                "mae": float(np.mean(np.abs(resid))),
                "r2": 1 - sse / sst if sst > 0 else np.nan}

    rows, all_resid = [], []
    for j in range(n_folds):
        te = row_fold == j
        res = sm.OLS(y[~te], X[~te]).fit()
        resid = y[te] - res.predict(X[te])
        moran = morans_i(resid, df)
        rows.append({"fold": j, "n_train": int((~te).sum()), **_metrics(resid),
                     "moran_i": moran["moran_i"], "moran_p": moran["moran_p"]})
        all_resid.append(resid)
    pooled = pd.concat(all_resid)  # every row held out exactly once
    pmoran = morans_i(pooled, df)
    rows.append({"fold": "pooled", "n_train": len(X), **_metrics(pooled),
                 "moran_i": pmoran["moran_i"], "moran_p": pmoran["moran_p"]})
    return pd.DataFrame(rows)


def print_cv_report(cv: pd.DataFrame) -> None:
    print(f"  {'fold':<7}{'n_test':>9}{'RMSE':>9}{'MAE':>9}{'R2':>9}"
          f"{'MoranI':>9}{'p':>8}")
    for _, r in cv.iterrows():
        mi = f"{r['moran_i']:9.4f}" if pd.notna(r["moran_i"]) else f"{'-':>9}"
        mp = f"{r['moran_p']:8.3g}" if pd.notna(r["moran_p"]) else f"{'-':>8}"
        print(f"  {str(r['fold']):<7}{int(r['n_test']):>9,}{r['rmse']:9.4f}"
              f"{r['mae']:9.4f}{r['r2']:9.4f}{mi}{mp}")


def fit_spec(df: pd.DataFrame, name: str, variables: list[str]) -> dict:
    X, y = build_design(df, variables)
    if len(X) == 0:
        raise ValueError(f"Spec '{name}' has no complete rows after dropna; "
                         "its variables are too sparse on this sample.")
    # Default fit uses a Moore-Penrose pseudoinverse, so any residual
    # collinearity among one-hot levels degrades gracefully instead of
    # raising a singular-matrix error.
    res = sm.OLS(y, X).fit()
    vif = compute_vif(X)
    resid = res.resid
    moran = morans_i(resid, df)
    return {
        "name": name,
        "result": res,
        "vif": vif,
        "n": int(res.nobs),
        "k": int(res.df_model),
        "rsq": res.rsquared,
        "rsq_adj": res.rsquared_adj,
        "aic": res.aic,
        "bic": res.bic,
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "mae": float(np.mean(np.abs(resid))),
        "f_stat": float(res.fvalue),
        "f_pvalue": float(res.f_pvalue),
        "durbin_watson": float(durbin_watson(resid)),
        **moran,
    }


def print_spec_report(fit: dict, top_vif: int = 15) -> None:
    res = fit["result"]
    print(f"\n{'=' * 72}\nSpec: {fit['name']}\n{'=' * 72}")
    print(f"n = {fit['n']:,}   k = {fit['k']}   "
          f"R^2 = {fit['rsq']:.4f}   adj R^2 = {fit['rsq_adj']:.4f}")
    print(f"AIC = {fit['aic']:,.1f}   BIC = {fit['bic']:,.1f}   "
          f"RMSE = {fit['rmse']:.4f}   MAE = {fit['mae']:.4f}")
    print(f"F({fit['k']}, {fit['n'] - fit['k'] - 1}) = {fit['f_stat']:,.1f}   "
          f"p = {fit['f_pvalue']:.3g}   Durbin-Watson = {fit['durbin_watson']:.3f}")
    print(f"Moran's I (residuals, {fit['moran_n']:,} parcels, kNN k={MORAN_K}): "
          f"{fit['moran_i']:.4f}   z = {fit['moran_z']:.1f}   "
          f"p = {fit['moran_p']:.3g}")
    print(f"\nTop {top_vif} VIF (year dummies excluded):")
    if fit["vif"].empty:
        print("  (no test variables)")
    else:
        for var, v in fit["vif"].head(top_vif).items():
            flag = "  <-- high" if v > 10 else ""
            print(f"  {var:<40} {v:10.1f}{flag}")


def save_outputs(fit: dict, out_dir: Path) -> None:
    """Write one spec's coefficient table, fit stats, and VIFs to CSV.

    Files are per spec (``hedonic_*_{spec}.csv``) so separate ``--spec`` runs
    accumulate instead of overwriting each other.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    name = fit["name"]

    res = fit["result"]
    ci = res.conf_int()
    coef = pd.DataFrame({
        "spec": name,
        "term": res.params.index,
        "coef": res.params.values,
        "std_err": res.bse.values,
        "t": res.tvalues.values,
        "p_value": res.pvalues.values,
        "ci_low": ci[0].values,
        "ci_high": ci[1].values,
    })
    coef_path = out_dir / f"hedonic_coefficients_{name}.csv"
    coef.to_csv(coef_path, index=False)

    stats = pd.DataFrame(
        [{"spec": name, "n": fit["n"], "k": fit["k"], "r2": fit["rsq"],
          "r2_adj": fit["rsq_adj"], "aic": fit["aic"], "bic": fit["bic"],
          "rmse": fit["rmse"], "mae": fit["mae"],
          "f_stat": fit["f_stat"], "f_pvalue": fit["f_pvalue"],
          "durbin_watson": fit["durbin_watson"],
          "moran_i": fit["moran_i"], "moran_z": fit["moran_z"],
          "moran_p": fit["moran_p"], "moran_n_parcels": fit["moran_n"]}])
    stats_path = out_dir / f"hedonic_fit_stats_{name}.csv"
    stats.to_csv(stats_path, index=False)

    vif_path = out_dir / f"hedonic_vif_{name}.csv"
    (fit["vif"].rename("vif").rename_axis("term").reset_index()
     .assign(spec=name)["spec term vif".split()]).to_csv(vif_path,
                                                          index=False)

    print(f"\nWrote spec '{name}' outputs:\n"
          f"  {coef_path}\n  {stats_path}\n  {vif_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Hedonic input parquet.")
    parser.add_argument("--spec", type=str, default=None,
                        help=f"Single spec to fit (one of {list(SPECS)}).")
    parser.add_argument("--sample", type=int, default=None,
                        help="Random-row sample size for fast iteration.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--spatial-folds", type=int, default=5,
                        help="K spatial folds for out-of-sample evaluation; "
                             "each fold holds out a contiguous band of parcels "
                             "within every neighborhood. 0 disables.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="Directory for coefficient / fit-stat / VIF CSVs.")
    args = parser.parse_args()
    args.input = require_existing_path(args.input, "Hedonic input parquet")
    return args


def main() -> None:
    args = parse_args()

    print(f"Reading hedonic input: {args.input}")
    df = pd.read_parquet(args.input)
    if args.sample and args.sample < len(df):
        df = df.sample(args.sample, random_state=args.seed)
        print(f"Sampled {len(df):,} rows (seed={args.seed})")

    specs = {args.spec: SPECS[args.spec]} if args.spec else SPECS
    fits = []
    for name, variables in specs.items():
        print(f"\nFitting spec '{name}' ({len(variables)} variables)...")
        fits.append(fit_spec(df, name, variables))
        print_spec_report(fits[-1])
        # Save per-spec files immediately so a stuck or crashed later spec
        # never loses the ones that already finished.
        save_outputs(fits[-1], args.out_dir)

    if len(fits) > 1:
        print(f"\n{'=' * 72}\nModel comparison (lower AIC/BIC is better)\n{'=' * 72}")
        comp = pd.DataFrame(
            {f["name"]: {"n": f["n"], "R2": round(f["rsq"], 4),
                         "AIC": round(f["aic"], 1), "BIC": round(f["bic"], 1)}
             for f in fits}
        ).T
        print(comp.to_string())

    if args.spatial_folds > 1:
        for name, variables in specs.items():
            print(f"\nSpatial CV for spec '{name}' "
                  f"({args.spatial_folds} folds, neighborhood-stratified)...")
            cv = spatial_cv(df, name, variables, args.spatial_folds,
                            seed=args.seed)
            cv.insert(0, "spec", name)
            print_cv_report(cv)
            cv_path = args.out_dir / f"hedonic_spatial_cv_{name}.csv"
            cv.to_csv(cv_path, index=False)
            print(f"\nWrote {cv_path}")


if __name__ == "__main__":
    main()
