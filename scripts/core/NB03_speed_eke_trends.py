#!/usr/bin/env python3
"""
NB03_speed_eke_trends.py  (CORRECTED)
======================================
Compute geostrophic speed and EKE per site, then run Sen's slope +
Mann-Kendall trend analysis for all meander metrics.

CORRECTED: EKE is now computed as spatial_mean(0.5*(u'² + v'²)) where
u' and v' are anomalies at EACH GRID POINT relative to the monthly
climatology at that grid point. The previous version incorrectly computed
anomalies of the spatial-mean velocity, yielding values ~1000x too small.

Requires: NB02 outputs (monthly_metrics_{site}.csv)
          NB02_patch_width.py should be run first to fix widths

NCI Gadi setup:
    source /g/data/gv90/xl1657/venvs/cmems_py311/bin/activate
    pip install pymannkendall   # only needed once
"""

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
import pymannkendall as mk
from statsmodels.tsa.stattools import acf
import warnings, gc

# =============================================================================
# 0. CONFIGURATION
# =============================================================================
BASE_DIR     = Path("/g/data/gv90/xl1657/cmems_adt")
ADT_FP       = BASE_DIR / "cmems_so30S_19930101_20250802_adt_sla_ugos_vgos_batch.nc"
PRODUCT_DIR  = BASE_DIR / "grl_meander_products"
OUT_DIR      = BASE_DIR / "grl_meander_products"

SITES = {
    "CP":   {"name": "Campbell Plateau",        "inner_lon": (150, -150), "inner_lat": (-57, -46), "wraps": True},
    "PAR":  {"name": "Pacific-Antarctic Ridge",  "inner_lon": (-150, -80), "inner_lat": (-60, -48), "wraps": False},
    "SEIR": {"name": "Southeast Indian Ridge",   "inner_lon": (130, 152), "inner_lat": (-56, -44), "wraps": False},
    "SWIR": {"name": "Southwest Indian Ridge",   "inner_lon": (15,  45),  "inner_lat": (-58, -44), "wraps": False},
}


# =============================================================================
# 1. COMPUTE SPEED AND EKE (CORRECTED)
# =============================================================================
def compute_speed_eke(adt_fp, sites, product_dir):
    """
    For each site, compute monthly geostrophic speed and EKE.

    Speed: spatial mean of daily |V_geo|, aggregated to monthly.
    EKE (CORRECTED): For each grid point, compute monthly-mean u and v,
    then monthly climatology, then anomalies. EKE at each grid point
    = 0.5*(u'² + v'²). The monthly EKE time series is the spatial mean
    of per-grid-point EKE values.
    """
    print("Computing speed and EKE from geostrophic velocities...")

    with Dataset(str(adt_fp), "r") as src:
        lat_name = "latitude" if "latitude" in src.variables else "lat"
        lon_name = "longitude" if "longitude" in src.variables else "lon"
        lat = np.array(src.variables[lat_name][:], dtype=np.float64)
        lon = np.array(src.variables[lon_name][:], dtype=np.float64)
        time_raw = src.variables["time"]
        time_all = pd.DatetimeIndex(
            xr.coding.times.decode_cf_datetime(
                time_raw[:], time_raw.units,
                calendar=getattr(time_raw, "calendar", "standard"),
            )
        )

        nt = len(time_all)
        TIME_CHUNK = 30

        for site_key, site in sites.items():
            print(f"\n  {site_key}: {site['name']}")

            ilon = site["inner_lon"]
            ilat_range = site["inner_lat"]
            wraps = site.get("wraps", False)
            jlat = np.where((lat >= ilat_range[0]) & (lat <= ilat_range[1]))[0]

            if not wraps:
                jlon = np.where((lon >= ilon[0]) & (lon <= ilon[1]))[0]
            else:
                jlon = np.where((lon >= ilon[0]) | (lon <= ilon[1]))[0]

            if len(jlat) == 0 or len(jlon) == 0:
                print(f"    WARNING: no grid points found. Skipping.")
                continue

            lat_s = slice(int(jlat[0]), int(jlat[-1]) + 1)
            nlat_site = int(jlat[-1]) - int(jlat[0]) + 1

            if not wraps or len(jlon) == 0:
                lon_s = slice(int(jlon[0]), int(jlon[-1]) + 1)
                lon_contiguous = True
                nlon_site = int(jlon[-1]) - int(jlon[0]) + 1
            else:
                gap = np.where(np.diff(jlon) > 1)[0]
                if len(gap) > 0:
                    split = gap[0] + 1
                    lon_s_1 = slice(int(jlon[:split][0]), int(jlon[:split][-1]) + 1)
                    lon_s_2 = slice(int(jlon[split:][0]), int(jlon[split:][-1]) + 1)
                    lon_contiguous = False
                    nlon_site = len(jlon)
                else:
                    lon_s = slice(int(jlon[0]), int(jlon[-1]) + 1)
                    lon_contiguous = True
                    nlon_site = int(jlon[-1]) - int(jlon[0]) + 1

            print(f"    Grid: {nlat_site} lat x {nlon_site} lon = {nlat_site * nlon_site} points")

            # ----------------------------------------------------------
            # Build month index for each day
            # ----------------------------------------------------------
            day_months = time_all.to_period("M")
            unique_months = day_months.unique()
            n_months = len(unique_months)
            month_dates = unique_months.to_timestamp() + pd.Timedelta(days=14)

            month_lookup = {m: i for i, m in enumerate(unique_months)}
            day_month_idx = np.array([month_lookup[m] for m in day_months])

            # Accumulators for monthly-mean u,v at each grid point
            u_month_sum = np.zeros((n_months, nlat_site, nlon_site), dtype=np.float64)
            v_month_sum = np.zeros((n_months, nlat_site, nlon_site), dtype=np.float64)
            month_count = np.zeros(n_months, dtype=np.int32)
            daily_speed = np.full(nt, np.nan)

            # ----------------------------------------------------------
            # Pass 1: Read daily data, accumulate monthly sums per grid point
            # ----------------------------------------------------------
            for cs in range(0, nt, TIME_CHUNK):
                ce = min(cs + TIME_CHUNK, nt)
                if lon_contiguous:
                    u = np.array(src.variables["ugos"][cs:ce, lat_s, lon_s], dtype=np.float32)
                    v = np.array(src.variables["vgos"][cs:ce, lat_s, lon_s], dtype=np.float32)
                else:
                    u = np.concatenate([
                        np.array(src.variables["ugos"][cs:ce, lat_s, lon_s_2], dtype=np.float32),
                        np.array(src.variables["ugos"][cs:ce, lat_s, lon_s_1], dtype=np.float32),
                    ], axis=2)
                    v = np.concatenate([
                        np.array(src.variables["vgos"][cs:ce, lat_s, lon_s_2], dtype=np.float32),
                        np.array(src.variables["vgos"][cs:ce, lat_s, lon_s_1], dtype=np.float32),
                    ], axis=2)

                u = np.where(np.abs(u) > 50, np.nan, u)
                v = np.where(np.abs(v) > 50, np.nan, v)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    spd = np.sqrt(u**2 + v**2)
                    daily_speed[cs:ce] = np.nanmean(spd, axis=(1, 2))

                for k in range(ce - cs):
                    day_idx = cs + k
                    mi = day_month_idx[day_idx]
                    u_day = np.where(np.isfinite(u[k]), u[k], 0.0)
                    v_day = np.where(np.isfinite(v[k]), v[k], 0.0)
                    u_month_sum[mi] += u_day
                    v_month_sum[mi] += v_day
                    month_count[mi] += 1

                if (cs // TIME_CHUNK) % 100 == 0 and cs > 0:
                    print(f"      Day {cs}/{nt}")

            # Monthly means at each grid point
            for mi in range(n_months):
                if month_count[mi] > 0:
                    u_month_sum[mi] /= month_count[mi]
                    v_month_sum[mi] /= month_count[mi]
                else:
                    u_month_sum[mi] = np.nan
                    v_month_sum[mi] = np.nan

            u_monthly = u_month_sum
            v_monthly = v_month_sum

            # ----------------------------------------------------------
            # Pass 2: Monthly climatology at each grid point
            # ----------------------------------------------------------
            cal_months = np.array([m.month for m in unique_months])
            u_clim = np.zeros((12, nlat_site, nlon_site), dtype=np.float64)
            v_clim = np.zeros((12, nlat_site, nlon_site), dtype=np.float64)
            clim_count = np.zeros(12, dtype=np.int32)

            for mi in range(n_months):
                cm = cal_months[mi] - 1
                u_clim[cm] += u_monthly[mi]
                v_clim[cm] += v_monthly[mi]
                clim_count[cm] += 1

            for cm in range(12):
                if clim_count[cm] > 0:
                    u_clim[cm] /= clim_count[cm]
                    v_clim[cm] /= clim_count[cm]

            # ----------------------------------------------------------
            # Pass 3: Anomalies → EKE at each grid point → spatial mean
            # ----------------------------------------------------------
            eke_monthly = np.full(n_months, np.nan)

            for mi in range(n_months):
                cm = cal_months[mi] - 1
                u_anom = u_monthly[mi] - u_clim[cm]
                v_anom = v_monthly[mi] - v_clim[cm]
                eke_field = 0.5 * (u_anom**2 + v_anom**2)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    eke_monthly[mi] = np.nanmean(eke_field)

            # Monthly speed from daily spatial means
            df_daily = pd.DataFrame({"speed": daily_speed}, index=time_all)
            speed_series = df_daily["speed"].resample("MS").mean()

            print(f"    Speed range: {speed_series.min():.3f}-{speed_series.max():.3f} m/s")
            print(f"    EKE range: {np.nanmin(eke_monthly):.6f}-{np.nanmax(eke_monthly):.6f} m2/s2")

            # ----------------------------------------------------------
            # Update CSV
            # ----------------------------------------------------------
            csv_fp = product_dir / f"monthly_metrics_{site_key}.csv"
            if csv_fp.exists():
                df_existing = pd.read_csv(csv_fp, index_col=0, parse_dates=True)
                df_existing["speed_m_s"] = np.nan
                df_existing["eke_m2_s2"] = np.nan

                speed_idx = speed_series.index + pd.Timedelta(days=14)
                for i, idx in enumerate(speed_idx):
                    if idx in df_existing.index and i < len(speed_series):
                        df_existing.loc[idx, "speed_m_s"] = speed_series.iloc[i]

                for mi in range(n_months):
                    md = month_dates[mi]
                    if md in df_existing.index:
                        df_existing.loc[md, "eke_m2_s2"] = eke_monthly[mi]

                df_existing.to_csv(csv_fp)
                print(f"    Updated: {csv_fp}")
            else:
                print(f"    WARNING: {csv_fp} not found. Run NB02 first.")

            del u_monthly, v_monthly, u_clim, v_clim, u_month_sum, v_month_sum
            gc.collect()


# =============================================================================
# 2. TREND ANALYSIS WITH PYMANNKENDALL
# =============================================================================
def compute_trends(product_dir, sites, alpha=0.05):
    """
    Compute Sen's slope and Mann-Kendall significance for all metrics.
    Uses modified MK (Hamed & Rao 1998) when lag-1 autocorrelation > 0.1.
    """
    print("\nComputing trends with Sen's slope + Mann-Kendall...")

    metrics = {
        "center_lat":    ("Position",       "deg/dec"),
        "width_km":      ("Width",          "km/dec"),
        "speed_m_s":     ("Speed",          "m s-1/dec"),
        "eke_m2_s2":     ("EKE",            "m2 s-2/dec"),
    }

    rows = []

    for site_key in sites:
        csv_fp = product_dir / f"monthly_metrics_{site_key}.csv"
        if not csv_fp.exists():
            print(f"  {site_key}: CSV not found, skipping.")
            continue

        df = pd.read_csv(csv_fp, index_col=0, parse_dates=True)
        print(f"\n  {site_key}: {sites[site_key]['name']}")

        for metric, (label, unit) in metrics.items():
            if metric not in df.columns:
                print(f"    {metric}: not in CSV, skipping.")
                continue

            y = df[metric].dropna()
            if len(y) < 24:
                print(f"    {metric}: too few points ({len(y)}), skipping.")
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

            from scipy import stats as sp_stats
            x_idx = np.arange(len(series))
            r_val = sp_stats.pearsonr(x_idx, series)[0]
            r2 = r_val**2

            sig_str = "***" if result.p < 0.01 else ("*" if result.p < 0.05 else "ns")

            rows.append({
                "site": site_key,
                "site_name": sites[site_key]["name"],
                "metric": metric,
                "metric_label": label,
                "unit": unit,
                "slope_per_decade": slope_per_decade,
                "p_value": result.p,
                "significant": result.p < alpha,
                "test": test_name,
                "acf_lag1": lag1,
                "R2": r2,
                "n_obs": len(series),
            })

            print(f"    {label:20s}  slope={slope_per_decade:+.4e}/{unit}  "
                  f"p={result.p:.4f}  {sig_str}  [{test_name}]")

    trends_df = pd.DataFrame(rows)

    out_fp = product_dir / "trend_results_mannkendall.csv"
    trends_df.to_csv(out_fp, index=False)
    print(f"\nTrend results saved to: {out_fp}")

    return trends_df


# =============================================================================
# 3. MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("GRL Speed, EKE & Trend Analysis Pipeline (CORRECTED)")
    print("  EKE = spatial_mean( 0.5*(u'^2 + v'^2) ) per grid point")
    print("=" * 70)

    compute_speed_eke(ADT_FP, SITES, PRODUCT_DIR)
    trends_df = compute_trends(PRODUCT_DIR, SITES)

    print("\n" + "=" * 70)
    print("TREND SUMMARY FOR GRL MANUSCRIPT")
    print("=" * 70)
    if len(trends_df) > 0:
        sig = trends_df[trends_df["significant"]]
        print(f"\nSignificant trends (p < 0.05): {len(sig)} of {len(trends_df)}")
        for _, row in sig.iterrows():
            print(f"  {row['site']:5s} {row['metric_label']:10s} "
                  f"{row['slope_per_decade']:+.4e} {row['unit']}  "
                  f"p={row['p_value']:.4f}")

    print("\nPipeline complete.")
