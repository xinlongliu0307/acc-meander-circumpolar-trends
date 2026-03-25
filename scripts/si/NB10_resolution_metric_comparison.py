#!/usr/bin/env python3
"""
NB10_resolution_metric_comparison.py
=====================================
Step 2 of GRL Supporting Information: Resolution and Metric Comparison.

Produces the data for Figure S2 + Table S2 by running meander detection
at CP and PAR under three conditions:

  Condition 1: 0.25° coarsened + 20% threshold + zero-crossing width
  Condition 2: 0.125° native   + 20% threshold + zero-crossing width
  Condition 3: 0.125° native   + 20% threshold + half-peak-height width
               (already computed — baseline from grl_meander_products)

Published values (from Liu et al. 2024, 2026) form a fourth reference row.

The zero-crossing width is the original width_deg output from NB02's
process_site() — i.e., the width BEFORE the half-peak-height patch.
The half-peak-height width is what's in the patched baseline CSVs.

Run on Gadi:
  source /g/data/gv90/xl1657/venvs/cmems_py311/bin/activate
  module purge
  unset PYTHONPATH
  cd /g/data/gv90/xl1657/cmems_adt/notebooks/
  python NB10_resolution_metric_comparison.py

Or submit as PBS:
  qsub run_resolution_comparison.sh

Author: Xinlong Liu, IMAS, University of Tasmania
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from netCDF4 import Dataset
import warnings, gc, time as _time, sys

warnings.filterwarnings("ignore")

# =============================================================================
# 0. CONFIGURATION
# =============================================================================

BASE_DIR = Path("/g/data/gv90/xl1657/cmems_adt")
ADT_FP   = BASE_DIR / "cmems_so30S_19930101_20250802_adt_sla_ugos_vgos_batch.nc"
PROD_DIR = BASE_DIR / "grl_meander_products"
COMP_DIR = PROD_DIR / "resolution_comparison"
COMP_DIR.mkdir(parents=True, exist_ok=True)

NOTEBOOK_DIR = BASE_DIR / "notebooks"

# Only CP and PAR (the two sites with published comparisons)
SITES_TO_COMPARE = ["CP", "PAR"]

# Published values from Liu et al. (2024) and Liu et al. (2026)
PUBLISHED = {
    "CP": {
        "source": "Liu et al. (2024)",
        "resolution": "0.25°",
        "product": "DT-2021",
        "threshold": "25%",
        "width_metric": "Zero-crossing",
        "period": "1993–2020",
        "position_trend": +0.12,
        "width_trend": +2.20,
        "speed_trend": +0.01,
        "position_sig": True,
        "width_sig": True,
        "speed_sig": True,
    },
    "PAR": {
        "source": "Liu et al. (2026)",
        "resolution": "0.25°",
        "product": "DT-2021",
        "threshold": "20%",
        "width_metric": "Zero-crossing",
        "period": "1993–2023",
        "position_trend": +0.03,
        "width_trend": +1.44,
        "speed_trend": +0.01,
        "position_sig": True,
        "width_sig": True,
        "speed_sig": True,
    },
}


# =============================================================================
# 1. COARSEN ADT DATA TO 0.25°
# =============================================================================

def create_coarsened_adt(adt_fp, out_fp):
    """
    Coarsen the 0.125° ADT data to 0.25° by 2×2 spatial averaging.
    Only processes adt variable to save time and memory.
    Writes a new NetCDF file.
    """
    if out_fp.exists():
        print(f"  Coarsened file already exists: {out_fp}")
        return

    print(f"  Creating coarsened 0.25° ADT file...")
    print(f"  Source: {adt_fp}")
    print(f"  Target: {out_fp}")

    t0 = _time.time()

    # Open with chunking to manage memory
    ds = xr.open_dataset(adt_fp, chunks={"time": 90})

    # Detect coordinate names
    lat_name = "latitude" if "latitude" in ds.dims else "lat"
    lon_name = "longitude" if "longitude" in ds.dims else "lon"

    # Verify grid spacing is 0.125°
    lat_vals = ds[lat_name].values
    grid_spacing = round(abs(float(lat_vals[1] - lat_vals[0])), 4)
    assert abs(grid_spacing - 0.125) < 0.01, \
        f"Expected 0.125° grid, got {grid_spacing}°"

    # Ensure even dimensions for coarsening
    nlat = len(ds[lat_name])
    nlon = len(ds[lon_name])
    if nlat % 2 != 0:
        ds = ds.isel({lat_name: slice(0, nlat - 1)})
    if nlon % 2 != 0:
        ds = ds.isel({lon_name: slice(0, nlon - 1)})

    # Coarsen: 2×2 spatial mean
    ds_coarse = ds[["adt"]].coarsen(
        {lat_name: 2, lon_name: 2}, boundary="trim"
    ).mean()

    # Also coarsen ugos and vgos if present (needed for speed)
    for var in ["ugos", "vgos"]:
        if var in ds:
            ds_coarse[var] = ds[var].coarsen(
                {lat_name: 2, lon_name: 2}, boundary="trim"
            ).mean()

    # Write to disk in chunks
    print(f"  Writing coarsened file (this may take several minutes)...")
    ds_coarse.to_netcdf(out_fp)
    ds.close()
    ds_coarse.close()

    elapsed = _time.time() - t0
    print(f"  Coarsened file created in {elapsed/60:.1f} min")
    print(f"  New grid: {grid_spacing*2:.3f}°")


# =============================================================================
# 2. RUN DETECTION AND COMPUTE TRENDS
# =============================================================================

def run_detection_and_trends(adt_fp, out_dir, label, sites=SITES_TO_COMPARE):
    """
    Run NB02 detection at the given ADT file, then compute trends.
    Returns a dict of {site: {metric: {slope, p_mod, sig}}}
    """
    # Import NB02
    import NB02_meander_detection as NB02
    import pymannkendall as mk

    results = {}

    for site_key in sites:
        csv_fp = out_dir / f"monthly_metrics_{site_key}.csv"

        if csv_fp.exists():
            print(f"  [{site_key}] {label}: already computed, loading CSV.")
        else:
            print(f"  [{site_key}] {label}: running detection...")
            t0 = _time.time()
            df, ds = NB02.process_site(
                site_key, adt_fp, out_dir,
                relat_thresh=0.20, x_months=4
            )
            del ds
            gc.collect()
            elapsed = _time.time() - t0
            print(f"  [{site_key}] Done in {elapsed/60:.1f} min")

        # Load CSV and compute trends
        df = pd.read_csv(csv_fp, parse_dates=["date"])
        df = df.dropna(subset=["center_lat"])

        site_results = {}
        for metric in ["center_lat", "width_km", "speed_m_s"]:
            if metric not in df.columns:
                site_results[metric] = {
                    "slope_dec": float("nan"),
                    "p_mod": float("nan"),
                    "sig": "—",
                }
                continue

            s = df[metric].dropna().values
            if len(s) < 24:
                site_results[metric] = {
                    "slope_dec": float("nan"),
                    "p_mod": float("nan"),
                    "sig": "—",
                }
                continue

            res = mk.original_test(s)
            try:
                mod = mk.hamed_rao_modification_test(s)
            except Exception:
                mod = res

            slope_dec = res.slope * 12 * 10
            p_mod = mod.p if mod else res.p
            sig = "**" if p_mod < 0.01 else ("*" if p_mod < 0.05 else "ns")

            site_results[metric] = {
                "slope_dec": slope_dec,
                "p_mod": p_mod,
                "sig": sig,
            }

            print(f"    {metric}: slope={slope_dec:.4f}/dec, p_mod={p_mod:.4f} ({sig})")

        results[site_key] = site_results

    return results


def apply_half_peak_height_and_trends(out_dir, sites=SITES_TO_COMPARE):
    """
    Apply half-peak-height width patch to detection outputs, then recompute
    width trends. Returns updated results dict.
    """
    import pymannkendall as mk

    # Import the patch function
    sys.path.insert(0, str(NOTEBOOK_DIR))
    from patch_width_all_thresholds import compute_half_peak_height_width

    results = {}

    for site_key in sites:
        nc_files = list(out_dir.glob(f"meander_detection_{site_key}_rel20_x4m.nc"))
        if not nc_files:
            print(f"  [{site_key}] NetCDF not found in {out_dir}, skipping.")
            results[site_key] = {}
            continue

        nc_fp = nc_files[0]
        csv_fp = out_dir / f"monthly_metrics_{site_key}.csv"

        # Load NetCDF
        ds = xr.open_dataset(nc_fp)
        ds.load()
        ds.close()

        occurrence = ds["occurrence"].values
        peak_freq = ds["peak_freq"].values
        lat = ds["latitude"].values

        # Compute half-peak-height width
        new_width_deg, _, _, n_comp, n_skip = \
            compute_half_peak_height_width(occurrence, lat, peak_freq)

        print(f"  [{site_key}] Half-peak-height: {n_comp} computed, {n_skip} skipped")

        # Compute site-averaged width
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_width_km = np.nanmean(new_width_deg, axis=1) * 111.32

        # Load CSV and replace width
        df = pd.read_csv(csv_fp, parse_dates=["date"])
        df = df.dropna(subset=["center_lat"])

        # Align lengths
        valid_mask = df["center_lat"].notna()
        if len(mean_width_km) == len(df):
            width_series = mean_width_km[valid_mask.values]
        else:
            width_series = mean_width_km[:valid_mask.sum()]

        # Compute width trend on half-peak-height
        s = pd.Series(width_series).dropna().values
        if len(s) >= 24:
            res = mk.original_test(s)
            try:
                mod = mk.hamed_rao_modification_test(s)
            except Exception:
                mod = res
            slope_dec = res.slope * 12 * 10
            p_mod = mod.p
            sig = "**" if p_mod < 0.01 else ("*" if p_mod < 0.05 else "ns")
        else:
            slope_dec, p_mod, sig = float("nan"), float("nan"), "—"

        results[site_key] = {
            "width_km": {"slope_dec": slope_dec, "p_mod": p_mod, "sig": sig},
        }
        print(f"    width_km (half-peak): slope={slope_dec:.4f}/dec, p_mod={p_mod:.4f} ({sig})")

    return results


# =============================================================================
# 3. MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("NB10: Resolution and Metric Comparison for GRL SI")
    print(f"  Sites: {SITES_TO_COMPARE}")
    print(f"  Output: {COMP_DIR}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # STEP A: Create coarsened 0.25° ADT file
    # ------------------------------------------------------------------
    print("\n>>> STEP A: Coarsen ADT data to 0.25°")
    coarse_fp = COMP_DIR / "cmems_coarsened_025deg.nc"
    create_coarsened_adt(ADT_FP, coarse_fp)

    # ------------------------------------------------------------------
    # STEP B: Condition 1 — 0.25° coarsened + zero-crossing width
    # ------------------------------------------------------------------
    print("\n>>> STEP B: Condition 1 — 0.25° + zero-crossing width")
    cond1_dir = COMP_DIR / "cond1_025deg_zerocross"
    cond1_dir.mkdir(parents=True, exist_ok=True)
    cond1_results = run_detection_and_trends(
        coarse_fp, cond1_dir, "0.25° zero-crossing"
    )

    # ------------------------------------------------------------------
    # STEP C: Condition 2 — 0.125° native + zero-crossing width
    # ------------------------------------------------------------------
    print("\n>>> STEP C: Condition 2 — 0.125° + zero-crossing width")
    cond2_dir = COMP_DIR / "cond2_0125deg_zerocross"
    cond2_dir.mkdir(parents=True, exist_ok=True)
    cond2_results = run_detection_and_trends(
        ADT_FP, cond2_dir, "0.125° zero-crossing"
    )

    # ------------------------------------------------------------------
    # STEP D: Condition 2b — apply half-peak-height to Condition 2 outputs
    # ------------------------------------------------------------------
    print("\n>>> STEP D: Condition 3 — 0.125° + half-peak-height width")
    print("  (Applying half-peak-height patch to Condition 2 outputs)")
    cond3_width = apply_half_peak_height_and_trends(cond2_dir)

    # ------------------------------------------------------------------
    # STEP E: Load baseline (Condition 3 from grl_meander_products)
    # ------------------------------------------------------------------
    print("\n>>> STEP E: Loading baseline results (0.125° + half-peak-height)")
    import pymannkendall as mk
    baseline_results = {}
    for site_key in SITES_TO_COMPARE:
        csv_fp = PROD_DIR / f"monthly_metrics_{site_key}.csv"
        df = pd.read_csv(csv_fp, parse_dates=["date"])
        df = df.dropna(subset=["center_lat"])
        site_res = {}
        for metric in ["center_lat", "width_km", "speed_m_s"]:
            s = df[metric].dropna().values
            if len(s) < 24:
                continue
            res = mk.original_test(s)
            try:
                mod = mk.hamed_rao_modification_test(s)
            except Exception:
                mod = res
            slope_dec = res.slope * 12 * 10
            p_mod = mod.p
            sig = "**" if p_mod < 0.01 else ("*" if p_mod < 0.05 else "ns")
            site_res[metric] = {"slope_dec": slope_dec, "p_mod": p_mod, "sig": sig}
        baseline_results[site_key] = site_res

    # ------------------------------------------------------------------
    # STEP F: Compile Table S2
    # ------------------------------------------------------------------
    print("\n>>> STEP F: Compiling Table S2")

    rows = []
    for site_key in SITES_TO_COMPARE:
        pub = PUBLISHED[site_key]

        # Row 1: Published
        rows.append({
            "site": site_key,
            "condition": "Published",
            "resolution": pub["resolution"],
            "product": pub["product"],
            "threshold": pub["threshold"],
            "width_metric": pub["width_metric"],
            "period": pub["period"],
            "position_slope": pub["position_trend"],
            "position_sig": "*" if pub["position_sig"] else "ns",
            "width_slope": pub["width_trend"],
            "width_sig": "*" if pub["width_sig"] else "ns",
            "speed_slope": pub["speed_trend"],
            "speed_sig": "*" if pub["speed_sig"] else "ns",
        })

        # Row 2: Cond 1 — 0.25° coarsened + zero-crossing
        c1 = cond1_results.get(site_key, {})
        rows.append({
            "site": site_key,
            "condition": "Cond 1",
            "resolution": "0.25°",
            "product": "DT-2024 coarsened",
            "threshold": "20%",
            "width_metric": "Zero-crossing",
            "period": "1993–2025",
            "position_slope": c1.get("center_lat", {}).get("slope_dec", np.nan),
            "position_sig": c1.get("center_lat", {}).get("sig", "—"),
            "width_slope": c1.get("width_km", {}).get("slope_dec", np.nan),
            "width_sig": c1.get("width_km", {}).get("sig", "—"),
            "speed_slope": c1.get("speed_m_s", {}).get("slope_dec", np.nan),
            "speed_sig": c1.get("speed_m_s", {}).get("sig", "—"),
        })

        # Row 3: Cond 2 — 0.125° + zero-crossing
        c2 = cond2_results.get(site_key, {})
        rows.append({
            "site": site_key,
            "condition": "Cond 2",
            "resolution": "0.125°",
            "product": "DT-2024",
            "threshold": "20%",
            "width_metric": "Zero-crossing",
            "period": "1993–2025",
            "position_slope": c2.get("center_lat", {}).get("slope_dec", np.nan),
            "position_sig": c2.get("center_lat", {}).get("sig", "—"),
            "width_slope": c2.get("width_km", {}).get("slope_dec", np.nan),
            "width_sig": c2.get("width_km", {}).get("sig", "—"),
            "speed_slope": c2.get("speed_m_s", {}).get("slope_dec", np.nan),
            "speed_sig": c2.get("speed_m_s", {}).get("sig", "—"),
        })

        # Row 4: Cond 3 — 0.125° + half-peak-height (baseline)
        bl = baseline_results.get(site_key, {})
        c3w = cond3_width.get(site_key, {})
        rows.append({
            "site": site_key,
            "condition": "Cond 3 (this study)",
            "resolution": "0.125°",
            "product": "DT-2024",
            "threshold": "20%",
            "width_metric": "Half-peak-height",
            "period": "1993–2025",
            "position_slope": bl.get("center_lat", {}).get("slope_dec", np.nan),
            "position_sig": bl.get("center_lat", {}).get("sig", "—"),
            "width_slope": bl.get("width_km", {}).get("slope_dec", np.nan),
            "width_sig": bl.get("width_km", {}).get("sig", "—"),
            "speed_slope": bl.get("speed_m_s", {}).get("slope_dec", np.nan),
            "speed_sig": bl.get("speed_m_s", {}).get("sig", "—"),
        })

    table_df = pd.DataFrame(rows)
    table_fp = COMP_DIR / "table_s2_resolution_comparison.csv"
    table_df.to_csv(table_fp, index=False)
    print(f"\nTable S2 saved: {table_fp}")
    print(table_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("DONE. All outputs in:", COMP_DIR)
    print("=" * 70)
