#!/usr/bin/env python3
"""
NB02_patch_width.py  (FIXED)
=============================
Recompute width from existing NB02 NetCDF outputs using half-peak-height
width, matching MATLAB findpeaks('WidthReference','halfheight').

NCI Gadi setup:
    source /g/data/gv90/xl1657/venvs/cmems_py311/bin/activate
"""

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import find_peaks
import warnings, shutil

PRODUCT_DIR = Path("/g/data/gv90/xl1657/cmems_adt/grl_meander_products")
MIN_PEAK_FREQ = 15
SITES = ["CP", "PAR", "SEIR", "SWIR"]


def half_peak_width(freq_profile, lat_1d, peak_idx):
    peak_val = freq_profile[peak_idx]
    if peak_val <= 0 or not np.isfinite(peak_val):
        return np.nan, np.nan, np.nan
    half_val = peak_val * 0.5
    n = len(freq_profile)
    south_lat = np.nan
    for j in range(peak_idx - 1, -1, -1):
        if freq_profile[j] <= half_val:
            f_lo, f_hi = freq_profile[j], freq_profile[j + 1]
            if f_hi - f_lo > 0:
                frac = (half_val - f_lo) / (f_hi - f_lo)
                south_lat = lat_1d[j] + frac * (lat_1d[j + 1] - lat_1d[j])
            else:
                south_lat = lat_1d[j]
            break
    north_lat = np.nan
    for j in range(peak_idx + 1, n):
        if freq_profile[j] <= half_val:
            f_hi, f_lo = freq_profile[j - 1], freq_profile[j]
            if f_hi - f_lo > 0:
                frac = (half_val - f_lo) / (f_hi - f_lo)
                north_lat = lat_1d[j] - frac * (lat_1d[j] - lat_1d[j - 1])
            else:
                north_lat = lat_1d[j]
            break
    if np.isfinite(south_lat) and np.isfinite(north_lat):
        return north_lat - south_lat, south_lat, north_lat
    return np.nan, np.nan, np.nan


def patch_site(site_key):
    nc_fp = PRODUCT_DIR / f"meander_detection_{site_key}_rel20_x4m.nc"
    csv_fp = PRODUCT_DIR / f"monthly_metrics_{site_key}.csv"
    if not nc_fp.exists():
        print(f"  {site_key}: NetCDF not found, skipping.")
        return

    ds = xr.open_dataset(nc_fp)
    lat_inner = ds["latitude"].values.copy()
    occurrence = ds["occurrence"].values.copy()
    peak_freq = ds["peak_freq"].values.copy()
    center_lat = ds["center_lat"].values.copy()
    month = pd.DatetimeIndex(ds["month"].values.copy())
    lon_inner = ds["longitude"].values.copy()
    all_vars = {v: ds[v].values.copy() for v in ds.data_vars}
    attrs = dict(ds.attrs)
    ds.close()

    n_months, n_lon = center_lat.shape
    old_mean = np.nanmean(all_vars["width_deg"]) * 111.32

    width_new = np.full((n_months, n_lon), np.nan)
    south_new = np.full((n_months, n_lon), np.nan)
    north_new = np.full((n_months, n_lon), np.nan)
    n_ok, n_skip = 0, 0

    for mi in range(n_months):
        for li in range(n_lon):
            pf = peak_freq[mi, li]
            if not np.isfinite(pf) or pf < MIN_PEAK_FREQ:
                n_skip += 1
                continue
            freq = occurrence[mi, :, li].astype(np.float64)
            if freq.max() <= 0:
                n_skip += 1
                continue
            pks_idx, props = find_peaks(freq, height=1)
            if len(pks_idx) == 0:
                n_skip += 1
                continue
            best = pks_idx[np.argmax(props["peak_heights"])]
            w, s, n = half_peak_width(freq, lat_inner, best)
            width_new[mi, li] = w
            south_new[mi, li] = s
            north_new[mi, li] = n
            n_ok += 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_w_deg = np.nanmean(width_new, axis=1)
        mean_w_km = mean_w_deg * 111.32
        mean_clat = np.nanmean(center_lat, axis=1)
        mean_pf = np.nanmean(peak_freq, axis=1)

    new_mean = np.nanmean(mean_w_km)
    print(f"  {site_key}: {n_ok} widths computed, {n_skip} skipped")
    print(f"    Width: {old_mean:.1f} km (old) -> {new_mean:.1f} km (half-height)")

    if csv_fp.exists():
        df = pd.read_csv(csv_fp, index_col=0, parse_dates=True)
    else:
        df = pd.DataFrame(index=month)
        df.index.name = "date"
    df["center_lat"] = mean_clat
    df["width_deg"] = mean_w_deg
    df["width_km"] = mean_w_km
    df["peak_frequency"] = mean_pf
    df.to_csv(csv_fp)
    print(f"    Updated CSV: {csv_fp}")

    all_vars["width_deg"] = width_new
    all_vars["south_edge"] = south_new
    all_vars["north_edge"] = north_new
    ds_out = xr.Dataset(
        {name: (["month", "latitude", "longitude"] if arr.ndim == 3
                else ["month", "longitude"], arr)
         for name, arr in all_vars.items()},
        coords={"month": month, "latitude": lat_inner, "longitude": lon_inner},
        attrs=attrs,
    )
    ds_out.attrs["width_definition"] = "half-peak-height (MATLAB findpeaks halfheight)"
    tmp = nc_fp.with_suffix(".tmp.nc")
    ds_out.to_netcdf(tmp)
    shutil.move(str(tmp), str(nc_fp))
    print(f"    Updated NetCDF: {nc_fp}")


if __name__ == "__main__":
    print("=" * 60)
    print("NB02 Width Patch (half-peak-height)")
    print(f"  Min peak frequency: {MIN_PEAK_FREQ} days")
    print("=" * 60)
    for s in SITES:
        patch_site(s)
    print("\nPatch complete.")
