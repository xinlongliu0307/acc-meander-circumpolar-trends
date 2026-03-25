#!/usr/bin/env python3
"""
NB05_era5_wind.py (REWRITTEN)
==============================
Analyse ERA5 10m wind data for GRL manuscript Section 3.4.

Reads the single downloaded file from NB05a_era5_download.py:
  /g/data/gv90/xl1657/era5_wind/era5_10m_wind_monthly_SO_1993_2025.nc

Extracts site-averaged monthly zonal wind and wind speed time series,
computes Mann-Kendall trends.

NCI Gadi setup:
    source /g/data/gv90/xl1657/venvs/cmems_py311/bin/activate
"""

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import pymannkendall as mk
from statsmodels.tsa.stattools import acf
import warnings

# =============================================================================
# 0. CONFIGURATION
# =============================================================================
ERA5_FP  = Path("/g/data/gv90/xl1657/era5_wind/era5_10m_wind_monthly_SO_1993_2025.nc")
OUT_DIR  = Path("/g/data/gv90/xl1657/cmems_adt/grl_meander_products")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SITES = {
    "CP":   {"name": "Campbell Plateau",       "lon": (150, -150), "lat": (-70, -30), "wraps": True},
    "PAR":  {"name": "Pacific-Antarctic Ridge", "lon": (-150, -60), "lat": (-70, -30), "wraps": False},
    "SEIR": {"name": "Southeast Indian Ridge",  "lon": (120, 155),  "lat": (-65, -30), "wraps": False},
    "SWIR": {"name": "Southwest Indian Ridge",  "lon": (5, 55),     "lat": (-65, -35), "wraps": False},
}


# =============================================================================
# 1. LOAD AND EXTRACT WIND PER SITE
# =============================================================================
def extract_wind(era5_fp, sites):
    """Load ERA5 file, extract site-averaged monthly wind for each site."""
    print("Loading ERA5 data...")

    ds = xr.open_dataset(era5_fp)
    print(f"  Variables: {list(ds.data_vars)}")
    print(f"  Dimensions: {dict(ds.sizes)}")

    # Identify variable names (CDS uses u10/v10 or 10u/10v)
    u_var = None
    v_var = None
    for vname in ds.data_vars:
        vlow = vname.lower()
        if vlow in ("u10", "10u", "u10m"):
            u_var = vname
        elif vlow in ("v10", "10v", "v10m"):
            v_var = vname
    if u_var is None or v_var is None:
        print(f"  Available variables: {list(ds.data_vars)}")
        # Try first two variables
        varnames = list(ds.data_vars)
        if len(varnames) >= 2:
            u_var, v_var = varnames[0], varnames[1]
            print(f"  Guessing: u={u_var}, v={v_var}")
        else:
            raise ValueError("Cannot identify u10 and v10 variables")

    print(f"  U variable: {u_var}")
    print(f"  V variable: {v_var}")

    # Identify coordinate names
    lat_name = None
    lon_name = None
    for cname in ds.coords:
        if "lat" in cname.lower():
            lat_name = cname
        if "lon" in cname.lower():
            lon_name = cname
    print(f"  Lat: {lat_name}, Lon: {lon_name}")

    lat_vals = ds[lat_name].values
    lon_vals = ds[lon_name].values
    lat_descending = lat_vals[0] > lat_vals[-1]

    print(f"  Lat range: {lat_vals.min():.1f} to {lat_vals.max():.1f}")
    print(f"  Lon range: {lon_vals.min():.1f} to {lon_vals.max():.1f}")

    # Check lon convention
    lon_is_360 = lon_vals.max() > 180

    site_data = {}

    for site_key, site in sites.items():
        print(f"\n  {site_key}: {site['name']}")

        lat_south, lat_north = site["lat"]
        lon_min, lon_max = site["lon"]
        wraps = site.get("wraps", False)

        # Select latitude (handle descending lat)
        if lat_descending:
            lat_sel = slice(lat_north, lat_south)
        else:
            lat_sel = slice(lat_south, lat_north)

        # Select longitude (handle convention and wrapping)
        if lon_is_360:
            # Convert -180/180 to 0/360
            lon_min_360 = lon_min % 360
            lon_max_360 = lon_max % 360
        else:
            lon_min_360 = lon_min
            lon_max_360 = lon_max

        if not wraps:
            u_site = ds[u_var].sel({lat_name: lat_sel, lon_name: slice(lon_min_360, lon_max_360)})
            v_site = ds[v_var].sel({lat_name: lat_sel, lon_name: slice(lon_min_360, lon_max_360)})
        else:
            # Dateline crossing
            if lon_is_360:
                # In 0-360: e.g., 150 to 210
                u_site = ds[u_var].sel({lat_name: lat_sel, lon_name: slice(lon_min_360, lon_max_360)})
                v_site = ds[v_var].sel({lat_name: lat_sel, lon_name: slice(lon_min_360, lon_max_360)})
            else:
                # In -180/180: select east + west
                u_east = ds[u_var].sel({lat_name: lat_sel, lon_name: slice(lon_min, 180)})
                u_west = ds[u_var].sel({lat_name: lat_sel, lon_name: slice(-180, lon_max)})
                u_site = xr.concat([u_east, u_west], dim=lon_name)
                v_east = ds[v_var].sel({lat_name: lat_sel, lon_name: slice(lon_min, 180)})
                v_west = ds[v_var].sel({lat_name: lat_sel, lon_name: slice(-180, lon_max)})
                v_site = xr.concat([v_east, v_west], dim=lon_name)

        # Spatial mean at each time step
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            u_ts = u_site.mean(dim=[lat_name, lon_name]).values
            v_ts = v_site.mean(dim=[lat_name, lon_name]).values

        # Build time index (CDS uses 'valid_time', standard is 'time')
        time_name = "valid_time" if "valid_time" in ds.coords else "time"
        time_vals = pd.DatetimeIndex(ds[time_name].values)

        df = pd.DataFrame({
            "u10": u_ts,
            "v10": v_ts,
        }, index=time_vals)
        df["wspd"] = np.sqrt(df["u10"]**2 + df["v10"]**2)

        print(f"    {len(df)} months ({df.index[0].date()} to {df.index[-1].date()})")
        print(f"    Mean zonal wind: {df['u10'].mean():.2f} m/s")
        print(f"    Mean wind speed: {df['wspd'].mean():.2f} m/s")

        site_data[site_key] = df

        # Save per-site CSV
        fp = OUT_DIR / f"era5_wind_{site_key}.csv"
        df.to_csv(fp)

    ds.close()
    return site_data


# =============================================================================
# 2. TREND ANALYSIS
# =============================================================================
def compute_wind_trends(site_data, alpha=0.05):
    """Compute Sen's slope + MK trends for zonal wind and wind speed."""
    print("\nComputing wind trends...")

    rows = []
    for site_key, df in site_data.items():
        if df is None:
            continue

        for metric, label, unit in [
            ("u10", "Zonal wind", "m s-1/dec"),
            ("wspd", "Wind speed", "m s-1/dec"),
        ]:
            y = df[metric].dropna()
            if len(y) < 24:
                continue

            y_anom = y - y.groupby(y.index.month).transform("mean")
            series = y_anom.values

            acf_vals = acf(series, nlags=5, fft=True)
            lag1 = acf_vals[1]
            use_modified = abs(lag1) > 0.1

            if use_modified:
                result = mk.hamed_rao_modification_test(series)
                test_name = "Modified MK"
            else:
                result = mk.original_test(series)
                test_name = "Original MK"

            slope_per_decade = result.slope * 12 * 10
            sig_str = "***" if result.p < 0.01 else ("*" if result.p < 0.05 else "ns")

            rows.append({
                "site": site_key,
                "site_name": SITES[site_key]["name"],
                "metric": metric,
                "metric_label": label,
                "unit": unit,
                "slope_per_decade": slope_per_decade,
                "p_value": result.p,
                "significant": result.p < alpha,
                "test": test_name,
            })

            print(f"  {site_key:5s} {label:15s}  slope={slope_per_decade:+.4f} {unit}  "
                  f"p={result.p:.4f}  {sig_str}")

    return pd.DataFrame(rows)


# =============================================================================
# 3. MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("GRL ERA5 Wind Analysis")
    print(f"  Input: {ERA5_FP}")
    print("=" * 60)

    if not ERA5_FP.exists():
        raise FileNotFoundError(
            f"ERA5 file not found: {ERA5_FP}\n"
            "Run NB05a_era5_download.py first to download the data."
        )

    site_data = extract_wind(ERA5_FP, SITES)

    trends_df = compute_wind_trends(site_data)

    out_fp = OUT_DIR / "era5_wind_trends.csv"
    trends_df.to_csv(out_fp, index=False)
    print(f"\nTrend results saved to: {out_fp}")

    # Summary
    print("\n" + "=" * 60)
    print("ERA5 WIND SUMMARY FOR GRL MANUSCRIPT")
    print("=" * 60)
    for _, row in trends_df[trends_df["metric"] == "u10"].iterrows():
        sig = "***" if row["p_value"] < 0.01 else ("*" if row["p_value"] < 0.05 else "ns")
        print(f"  {row['site']:5s} Zonal wind: {row['slope_per_decade']:+.3f} m/s/dec  "
              f"p={row['p_value']:.4f} {sig}")

    print("\nERA5 analysis complete.")
