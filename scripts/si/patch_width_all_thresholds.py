#!/usr/bin/env python3
"""
patch_width_all_thresholds.py (v2 — fixed NetCDF write)
========================================================
Apply the half-peak-height width correction to all threshold sensitivity
outputs so that width trends are comparable across thresholds.

The baseline 20% outputs were already patched by NB02_patch_width.py.
This script applies the same correction to the 15%, 25%, 30%, 35%, 40%
outputs in threshold_sensitivity/relXX/.

Fix in v2: load dataset fully into memory before closing and rewriting.

Run on Gadi:
  source /g/data/gv90/xl1657/venvs/cmems_py311/bin/activate
  module purge
  unset PYTHONPATH
  cd /g/data/gv90/xl1657/cmems_adt/notebooks/
  python patch_width_all_thresholds.py
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import warnings

SENS_DIR = Path("/g/data/gv90/xl1657/cmems_adt/grl_meander_products/threshold_sensitivity")
SITES = ["CP", "PAR", "SEIR", "SWIR"]
THRESHOLDS_TO_PATCH = [15, 25, 30, 35, 40]  # 20% already patched
MIN_PEAK_FREQ = 15  # days, same as NB02_patch_width.py


def compute_half_peak_height_width(occurrence_3d, lat_1d, peak_freq_2d,
                                   min_peak_freq=MIN_PEAK_FREQ):
    """
    Recompute meander width using half-peak-height definition.

    For each month and longitude, find the latitude of peak occurrence,
    then find the north and south latitudes where occurrence drops to
    half the peak value. Width = north_half - south_half.
    """
    n_months, n_lat, n_lon = occurrence_3d.shape
    width_deg  = np.full((n_months, n_lon), np.nan)
    south_edge = np.full((n_months, n_lon), np.nan)
    north_edge = np.full((n_months, n_lon), np.nan)
    n_computed = 0
    n_skipped  = 0

    for m in range(n_months):
        for j in range(n_lon):
            freq = peak_freq_2d[m, j]
            if np.isnan(freq) or freq < min_peak_freq:
                n_skipped += 1
                continue

            profile = occurrence_3d[m, :, j]
            if np.all(np.isnan(profile)) or np.nanmax(profile) == 0:
                n_skipped += 1
                continue

            peak_val = np.nanmax(profile)
            half_val = peak_val / 2.0
            peak_idx = np.nanargmax(profile)

            # Search southward for half-height crossing
            south_idx = peak_idx
            for k in range(peak_idx - 1, -1, -1):
                if np.isnan(profile[k]):
                    continue
                if profile[k] <= half_val:
                    if not np.isnan(profile[k + 1]) and profile[k + 1] != profile[k]:
                        frac = (half_val - profile[k]) / (profile[k + 1] - profile[k])
                        south_idx = k + frac
                    else:
                        south_idx = k
                    break
            else:
                south_idx = 0

            # Search northward for half-height crossing
            north_idx = peak_idx
            for k in range(peak_idx + 1, n_lat):
                if np.isnan(profile[k]):
                    continue
                if profile[k] <= half_val:
                    if not np.isnan(profile[k - 1]) and profile[k - 1] != profile[k]:
                        frac = (half_val - profile[k]) / (profile[k - 1] - profile[k])
                        north_idx = k - frac
                    else:
                        north_idx = k
                    break
            else:
                north_idx = n_lat - 1

            s_lat = np.interp(south_idx, np.arange(n_lat), lat_1d)
            n_lat_val = np.interp(north_idx, np.arange(n_lat), lat_1d)

            width_deg[m, j]  = abs(n_lat_val - s_lat)
            south_edge[m, j] = min(s_lat, n_lat_val)
            north_edge[m, j] = max(s_lat, n_lat_val)
            n_computed += 1

    return width_deg, south_edge, north_edge, n_computed, n_skipped


def patch_one_threshold(thresh_pct):
    """Apply half-peak-height width patch to one threshold directory."""
    thresh_dir = SENS_DIR / f"rel{thresh_pct:02d}"
    print(f"\n{'='*60}")
    print(f"Patching threshold {thresh_pct}%")
    print(f"{'='*60}")

    for site_key in SITES:
        nc_files = list(thresh_dir.glob(f"meander_detection_{site_key}_rel{thresh_pct}_x4m.nc"))
        if not nc_files:
            print(f"  {site_key}: NetCDF not found, skipping.")
            continue

        nc_fp  = nc_files[0]
        csv_fp = thresh_dir / f"monthly_metrics_{site_key}.csv"

        # ---- Load EVERYTHING into memory, then close the file ----
        ds = xr.open_dataset(nc_fp)
        ds.load()                       # force all data into RAM
        ds.close()                      # release the file handle

        occurrence = ds["occurrence"].values     # (month, lat, lon)
        peak_freq  = ds["peak_freq"].values      # (month, lon)
        lat        = ds["latitude"].values
        old_width  = ds["width_deg"].values

        old_mean_km = np.nanmean(old_width) * 111.32

        # Compute half-peak-height width
        new_width_deg, new_south, new_north, n_comp, n_skip = \
            compute_half_peak_height_width(occurrence, lat, peak_freq)

        new_mean_km = np.nanmean(new_width_deg) * 111.32

        print(f"  {site_key}: {n_comp} widths computed, {n_skip} skipped")
        print(f"    Width: {old_mean_km:.1f} km (old) -> {new_mean_km:.1f} km (half-height)")

        # ---- Write updated NetCDF (file is already closed) ----
        ds["width_deg"].values  = new_width_deg
        ds["south_edge"].values = new_south
        ds["north_edge"].values = new_north
        ds.attrs["width_definition"] = "half-peak-height"
        ds.to_netcdf(nc_fp, mode="w")
        print(f"    Updated NetCDF: {nc_fp}")

        # ---- Update CSV ----
        df = pd.read_csv(csv_fp, parse_dates=["date"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_width_deg = np.nanmean(new_width_deg, axis=1)
        mean_width_km = mean_width_deg * 111.32

        if len(df) == len(mean_width_km):
            df["width_deg"] = mean_width_deg
            df["width_km"]  = mean_width_km
            df.to_csv(csv_fp, index=False)
            print(f"    Updated CSV: {csv_fp}")
        else:
            print(f"    WARNING: CSV rows ({len(df)}) != NetCDF months ({len(mean_width_km)}). CSV not updated.")

    print(f"  Threshold {thresh_pct}% done.")


if __name__ == "__main__":
    print("=" * 60)
    print("Half-peak-height width patch (v2) for threshold sensitivity")
    print(f"  Thresholds to patch: {THRESHOLDS_TO_PATCH}")
    print(f"  Min peak frequency: {MIN_PEAK_FREQ} days")
    print("=" * 60)

    for thresh in THRESHOLDS_TO_PATCH:
        patch_one_threshold(thresh)

    print(f"\n{'='*60}")
    print("All thresholds patched.")
    print("Now re-run: python compute_sensitivity_trends.py")
    print(f"{'='*60}")
