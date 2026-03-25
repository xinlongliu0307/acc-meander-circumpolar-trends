#!/usr/bin/env python3
"""
NB04_argo_temperature.py
========================
Analyse Roemmich-Gilson Argo temperature for GRL manuscript Section 3.4.

Steps:
  1. Load base climatology (2004-2018) + monthly extensions (2019-2025)
  2. Extract temperature at ~500 dbar for each site
  3. Compute two decadal averages: 2006-2015 and 2016-2025
  4. Compute meridional temperature gradient at 500 m
  5. Map the change in gradient between decades
  6. Save results for manuscript

Data location on Gadi:
  /g/data/gv90/xl1657/argo_rg/RG_ArgoClim_Temperature_2019.nc   (base)
  /g/data/gv90/xl1657/argo_rg/RG_ArgoClim_YYYYMM_2019.nc       (extensions)

NCI Gadi setup:
    source /g/data/gv90/xl1657/venvs/cmems_py311/bin/activate
"""

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import warnings
import glob

# =============================================================================
# 0. CONFIGURATION
# =============================================================================
ARGO_DIR = Path("/g/data/gv90/xl1657/argo_rg")
OUT_DIR  = Path("/g/data/gv90/xl1657/cmems_adt/grl_meander_products")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Target pressure level (~500 m depth)
TARGET_PRES = 500.0  # dbar

# Site outer domains (for temperature analysis — use broader region than inner)
# Longitude in -180/180 convention
SITES = {
    "CP":   {"name": "Campbell Plateau",       "lon": (150, -150), "lat": (-70, -30), "wraps": True},
    "PAR":  {"name": "Pacific-Antarctic Ridge", "lon": (-150, -60), "lat": (-70, -30), "wraps": False},
    "SEIR": {"name": "Southeast Indian Ridge",  "lon": (120, 155),  "lat": (-65, -30), "wraps": False},
    "SWIR": {"name": "Southwest Indian Ridge",  "lon": (5, 55),     "lat": (-65, -35), "wraps": False},
}

# Decades for comparison
DECADE_EARLY = ("2006-01", "2015-12")
DECADE_LATE  = ("2016-01", "2025-12")


# =============================================================================
# 1. LOAD ARGO DATA
# =============================================================================
def load_argo_data(argo_dir):
    """
    Load the base climatology and all monthly extensions.
    Returns a single DataArray of temperature anomalies with time dimension.
    """
    print("Loading Argo data...")

    # Base climatology
    base_fp = argo_dir / "RG_ArgoClim_Temperature_2019.nc"
    if not base_fp.exists():
        raise FileNotFoundError(f"Base climatology not found: {base_fp}")

    ds_base = xr.open_dataset(base_fp, decode_times=False)
    print(f"  Base climatology: {base_fp.name}")
    print(f"  Variables: {list(ds_base.data_vars)}")
    print(f"  Dimensions: {dict(ds_base.sizes)}")

    # Identify the temperature anomaly variable
    # Common names: ARGO_TEMPERATURE_ANOMALY, TEMP_ANOMALY, argo_temperature_anomaly
    temp_var = None
    for vname in ds_base.data_vars:
        if "TEMP" in vname.upper() and "ANOM" in vname.upper():
            temp_var = vname
            break
    if temp_var is None:
        # Try the mean field
        for vname in ds_base.data_vars:
            if "TEMP" in vname.upper() and "MEAN" in vname.upper():
                temp_var = vname
                break
    if temp_var is None:
        print(f"  Available variables: {list(ds_base.data_vars)}")
        raise ValueError("Cannot find temperature variable in base climatology")

    print(f"  Using variable: {temp_var}")

    # Get pressure/depth coordinate name
    pres_name = None
    for cname in ds_base.coords:
        if "PRES" in cname.upper() or "DEPTH" in cname.upper() or "LEVEL" in cname.upper():
            pres_name = cname
            break
    if pres_name is None:
        for dname in ds_base.dims:
            if "PRES" in dname.upper() or "DEPTH" in dname.upper():
                pres_name = dname
                break

    print(f"  Pressure coordinate: {pres_name}")

    # Get lat/lon coordinate names
    lat_name = None
    lon_name = None
    for cname in list(ds_base.coords) + list(ds_base.dims):
        if "LAT" in cname.upper():
            lat_name = cname
        if "LON" in cname.upper():
            lon_name = cname

    print(f"  Lat: {lat_name}, Lon: {lon_name}")

    # Base climatology has mean + annual cycle in the time dimension
    # TIME dimension = 180 months = Jan 2004 to Dec 2018
    # Time is "months since 2004-01-01" so index 0=Jan 2004, 1=Feb 2004, etc.
    base_temp = ds_base[temp_var]
    print(f"  Base shape: {base_temp.shape}")

    time_dim = None
    for d in base_temp.dims:
        if "TIME" in d.upper():
            time_dim = d
            break
    n_base_months = base_temp.sizes[time_dim]

    # Build dates for base climatology: Jan 2004 + i months
    base_dates = [pd.Timestamp("2004-01-15") + pd.DateOffset(months=i)
                  for i in range(n_base_months)]
    print(f"  Base time range: {base_dates[0].date()} to {base_dates[-1].date()} "
          f"({n_base_months} months)")

    # Collect ALL monthly fields: base (2004-2018) + extensions (2019-2025)
    monthly_temps = []
    monthly_dates = []

    # Extract each month from base climatology
    for i in range(n_base_months):
        monthly_temps.append(base_temp.isel({time_dim: i}))
        monthly_dates.append(base_dates[i])

    print(f"  Extracted {n_base_months} months from base climatology")

    # Load monthly extensions (2019 onwards)
    ext_files = sorted(argo_dir.glob("RG_ArgoClim_20*_2019.nc"))
    # Exclude the base climatology file itself
    ext_files = [fp for fp in ext_files if "Temperature" not in fp.name
                 and "Salinity" not in fp.name and "mean" not in fp.name
                 and "annual" not in fp.name]
    print(f"  Found {len(ext_files)} monthly extension files")

    n_ext = 0
    for fp in ext_files:
        fname = fp.stem  # e.g., RG_ArgoClim_202301_2019
        parts = fname.split("_")
        for p in parts:
            if len(p) == 6 and p.isdigit():
                year = int(p[:4])
                month = int(p[4:6])
                if 2019 <= year <= 2026 and 1 <= month <= 12:
                    try:
                        ds_ext = xr.open_dataset(fp, decode_times=False)
                        ext_var = None
                        for vname in ds_ext.data_vars:
                            if "TEMP" in vname.upper() and "ANOM" in vname.upper():
                                ext_var = vname
                                break
                        if ext_var is None:
                            for vname in ds_ext.data_vars:
                                if "TEMP" in vname.upper():
                                    ext_var = vname
                                    break
                        if ext_var is not None:
                            da = ds_ext[ext_var]
                            # Remove time dim if present (extensions have 1 time step)
                            for d in da.dims:
                                if "TIME" in d.upper():
                                    da = da.isel({d: 0})
                                    break
                            monthly_temps.append(da)
                            monthly_dates.append(pd.Timestamp(year=year, month=month, day=15))
                            n_ext += 1
                        ds_ext.close()
                    except Exception as e:
                        print(f"    Warning: could not read {fp.name}: {e}")
                break

    print(f"  Loaded {n_ext} monthly extension fields")
    print(f"  Total monthly fields: {len(monthly_temps)} "
          f"({monthly_dates[0].date()} to {monthly_dates[-1].date()})")

    ds_base.close()

    return base_temp, monthly_temps, monthly_dates, pres_name, lat_name, lon_name


# =============================================================================
# 2. ANALYSE TEMPERATURE AT EACH SITE
# =============================================================================
def analyse_site(site_key, site, base_temp, monthly_temps, monthly_dates,
                 pres_name, lat_name, lon_name, target_pres=TARGET_PRES):
    """
    For a single site:
      1. Subset to site domain at target pressure
      2. Compute decadal averages
      3. Compute meridional temperature gradient
      4. Compute gradient change between decades
    """
    print(f"\n  {site_key}: {site['name']}")

    # Find nearest pressure level to target
    if pres_name is not None and pres_name in base_temp.coords:
        pres_vals = base_temp[pres_name].values
        pres_idx = np.argmin(np.abs(pres_vals - target_pres))
        actual_pres = pres_vals[pres_idx]
        print(f"    Target pressure: {target_pres} dbar, nearest: {actual_pres} dbar")
    else:
        print(f"    WARNING: cannot find pressure coordinate, using index 25")
        pres_idx = 25
        actual_pres = target_pres

    # Subset monthly extensions to site domain at target pressure
    early_temps = []
    late_temps = []

    early_start = pd.Timestamp(DECADE_EARLY[0])
    early_end = pd.Timestamp(DECADE_EARLY[1])
    late_start = pd.Timestamp(DECADE_LATE[0])
    late_end = pd.Timestamp(DECADE_LATE[1])

    for i, date in enumerate(monthly_dates):
        temp = monthly_temps[i]

        # Select pressure level
        if pres_name is not None and pres_name in temp.dims:
            temp_slice = temp.isel({pres_name: pres_idx})
        elif pres_name is not None and pres_name in temp.coords:
            temp_slice = temp.sel({pres_name: actual_pres}, method="nearest")
        else:
            temp_slice = temp.isel({list(temp.dims)[0]: pres_idx}) if len(temp.dims) > 2 else temp

        # Remove time dimension if present
        if "time" in temp_slice.dims or "TIME" in temp_slice.dims:
            tdim = "time" if "time" in temp_slice.dims else "TIME"
            temp_slice = temp_slice.isel({tdim: 0})

        # Subset to site domain
        lat_vals = temp_slice[lat_name].values if lat_name in temp_slice.coords else None
        lon_vals = temp_slice[lon_name].values if lon_name in temp_slice.coords else None

        if lat_vals is not None and lon_vals is not None:
            lat_mask = (lat_vals >= site["lat"][0]) & (lat_vals <= site["lat"][1])

            # Handle longitude convention: Argo uses 0-360, sites use -180/180
            lon_min, lon_max = site["lon"]
            if lon_vals.min() >= 0 and lon_vals.max() > 180:
                # Data is in 0-360 convention — convert site lons
                lon_min_360 = lon_min % 360
                lon_max_360 = lon_max % 360
                if not site.get("wraps", False):
                    lon_mask = (lon_vals >= lon_min_360) & (lon_vals <= lon_max_360)
                else:
                    # Wraps: e.g., 150 to 210 in 0-360
                    lon_mask = (lon_vals >= lon_min_360) | (lon_vals <= lon_max_360)
            else:
                # Data is in -180/180
                if not site.get("wraps", False):
                    lon_mask = (lon_vals >= lon_min) & (lon_vals <= lon_max)
                else:
                    lon_mask = (lon_vals >= lon_min) | (lon_vals <= lon_max)

            temp_site = temp_slice.isel(
                {lat_name: np.where(lat_mask)[0],
                 lon_name: np.where(lon_mask)[0]}
            )
        else:
            temp_site = temp_slice

        # Assign to decade
        if early_start <= date <= early_end:
            early_temps.append(temp_site.values)
        elif late_start <= date <= late_end:
            late_temps.append(temp_site.values)

    if len(early_temps) == 0 or len(late_temps) == 0:
        print(f"    WARNING: insufficient data (early={len(early_temps)}, late={len(late_temps)})")
        return None

    print(f"    Early decade: {len(early_temps)} months, Late decade: {len(late_temps)} months")

    # Compute decadal means
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        T_early = np.nanmean(np.array(early_temps), axis=0)
        T_late = np.nanmean(np.array(late_temps), axis=0)
        T_diff = T_late - T_early

    # Compute meridional temperature gradient at target depth
    # dT/dy in °C per degree latitude
    if lat_vals is not None:
        lat_site = lat_vals[lat_mask]
        dlat = np.mean(np.diff(lat_site))
        dT_dy_early = np.gradient(T_early, dlat, axis=0)  # °C per degree
        dT_dy_late = np.gradient(T_late, dlat, axis=0)
        dT_dy_change = dT_dy_late - dT_dy_early

        # Convert to °C per 100 km
        dT_dy_early_100km = dT_dy_early / 111.32 * 100
        dT_dy_late_100km = dT_dy_late / 111.32 * 100
        dT_dy_change_100km = dT_dy_change / 111.32 * 100
    else:
        dT_dy_change_100km = T_diff * 0  # placeholder

    # Summary statistics
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_early = np.nanmean(T_early)
        mean_late = np.nanmean(T_late)
        mean_diff = np.nanmean(T_diff)
        mean_grad_change = np.nanmean(np.abs(dT_dy_change_100km))

    print(f"    Mean T (early): {mean_early:.3f}°C anomaly")
    print(f"    Mean T (late):  {mean_late:.3f}°C anomaly")
    print(f"    Mean warming:   {mean_diff:+.3f}°C")
    print(f"    Mean |grad change|: {mean_grad_change:.4f} °C/100km")

    return {
        "T_early": T_early,
        "T_late": T_late,
        "T_diff": T_diff,
        "dT_dy_change_100km": dT_dy_change_100km,
        "mean_warming": mean_diff,
        "mean_grad_change": mean_grad_change,
        "n_early": len(early_temps),
        "n_late": len(late_temps),
    }


# =============================================================================
# 3. MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("GRL Argo Temperature Analysis")
    print(f"  Target depth: {TARGET_PRES} dbar")
    print(f"  Early decade: {DECADE_EARLY}")
    print(f"  Late decade:  {DECADE_LATE}")
    print("=" * 60)

    base_temp, monthly_temps, monthly_dates, pres_name, lat_name, lon_name = \
        load_argo_data(ARGO_DIR)

    results = {}
    for site_key, site in SITES.items():
        result = analyse_site(
            site_key, site, base_temp, monthly_temps, monthly_dates,
            pres_name, lat_name, lon_name
        )
        if result is not None:
            results[site_key] = result

    # Summary table
    print("\n" + "=" * 60)
    print("ARGO SUMMARY FOR GRL MANUSCRIPT (Section 3.4)")
    print("=" * 60)
    print(f"{'Site':6s} {'Warming (°C)':>14s} {'|Grad change|':>14s} {'N_early':>8s} {'N_late':>8s}")
    print("-" * 56)
    for key, r in results.items():
        print(f"{key:6s} {r['mean_warming']:+14.4f} {r['mean_grad_change']:14.4f} "
              f"{r['n_early']:8d} {r['n_late']:8d}")

    # Save summary CSV
    rows = []
    for key, r in results.items():
        rows.append({
            "site": key,
            "site_name": SITES[key]["name"],
            "mean_warming_C": r["mean_warming"],
            "mean_abs_grad_change_C_100km": r["mean_grad_change"],
            "n_months_early": r["n_early"],
            "n_months_late": r["n_late"],
        })
    df = pd.DataFrame(rows)
    out_fp = OUT_DIR / "argo_temperature_summary.csv"
    df.to_csv(out_fp, index=False)
    print(f"\nSaved: {out_fp}")
    print("\nArgo analysis complete.")
