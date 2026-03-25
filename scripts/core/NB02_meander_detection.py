#!/usr/bin/env python3
"""
NB02_meander_detection.py
=========================
Core meander detection algorithm adapted from:
  f_derive_meander_location_daily.m  (Amelie Meyer, June 2020; Xinlong Liu, Sep 2021)

Adapted for:
  GRL manuscript "Circumpolar Satellite Evidence for Topographically-Modulated
  Multi-Decadal Evolution of Southern Ocean Standing Meanders" (Liu, 2026)

Specifications from manuscript:
  - CMEMS ADT 0.125° DT-2024 (Jan 1993 – Aug 2025)
  - 20% relative threshold of local domain maximum per day
  - 4-month aggregation window
  - Zero-frequency crossing width definition
  - Four sites: CP, PAR, SEIR, SWIR

Run on NCI Gadi via JupyterLab (ARE) or as:
  python NB02_meander_detection.py

Author: Xinlong Liu, IMAS, University of Tasmania

NCI Gadi setup (run once per session before launching this script):
    source /g/data/gv90/xl1657/venvs/cmems_py311/bin/activate
    pip install pymannkendall   # only needed once
"""

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
from scipy.signal import find_peaks
import warnings, gc, time as _time

# =============================================================================
# 0. CONFIGURATION
# =============================================================================

# --- File paths (NCI Gadi) ---
BASE_DIR = Path("/g/data/gv90/xl1657/cmems_adt")
# The concatenated CMEMS ADT file covering Jan 1993 – Aug 2025.
# Adjust filename if your file differs.
ADT_FP = BASE_DIR / "cmems_so30S_19930101_20250802_adt_sla_ugos_vgos_batch.nc"
OUT_DIR = BASE_DIR / "grl_meander_products"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Algorithm parameters (matching GRL manuscript) ---
RELAT_THRESH = 0.20          # 20% of local domain maximum per day
X_MONTHS     = 4             # 4-month aggregation window (manuscript Sec 2.2)
EARTH_RADIUS = 6371.0e3      # m
DEG2RAD      = np.pi / 180.0

# --- Four study sites (from manuscript Table in Sec 2.2) ---
# Outer domain: for ADT gradient computation
# Inner domain: for meander tracking
# Longitudes in degrees East matching the CMEMS file convention (-180 to 180).
# CP and PAR cross the dateline or lie in the Western Hemisphere:
#   Manuscript 210°E = -150°, 280°E = -80°, 300°E = -60°
SITES = {
    "CP": {
        "name": "Campbell Plateau",
        "outer_lon": (150, -150), "outer_lat": (-70, -30),   # crosses dateline
        "inner_lon": (150, -150), "inner_lat": (-57, -46),
        "wraps": True,
        "color": "#1f77b4",
    },
    "PAR": {
        "name": "Pacific-Antarctic Ridge",
        "outer_lon": (-150, -60), "outer_lat": (-70, -30),
        "inner_lon": (-150, -80), "inner_lat": (-60, -48),
        "wraps": False,
        "color": "#d62728",
    },
    "SEIR": {
        "name": "Southeast Indian Ridge",
        "outer_lon": (120, 155), "outer_lat": (-65, -30),
        "inner_lon": (130, 152), "inner_lat": (-56, -44),
        "wraps": False,
        "color": "#2ca02c",
    },
    "SWIR": {
        "name": "Southwest Indian Ridge",
        "outer_lon": (5, 55),   "outer_lat": (-65, -35),
        "inner_lon": (15, 45),  "inner_lat": (-58, -44),
        "wraps": False,
        "color": "#ff7f0e",
    },
}


def select_lon_indices(lon_full, lon_min, lon_max, wraps):
    """
    Select longitude indices, handling dateline crossing.

    For non-wrapping: lon_min < lon_max, simple range.
    For wrapping (e.g., 150 to -150): select lon >= 150 OR lon <= -150.
    Returns sorted indices into lon_full.
    """
    if not wraps:
        return np.where((lon_full >= lon_min) & (lon_full <= lon_max))[0]
    else:
        # Wraps across the dateline: select both sides
        idx = np.where((lon_full >= lon_min) | (lon_full <= lon_max))[0]
        return idx


# =============================================================================
# 1. HELPER: COMPUTE ADT GRADIENT MAGNITUDE
# =============================================================================
# Faithfully adapted from MATLAB lines 66-79:
#   [dx,dy] = gradient(alti_ADT(:,:,i), 0.25);
#   dx_ADT  = dx ./ (pi/180 * Rearth * cos(deg2rad(alti_lat)));
#   dy_ADT  = dy ./ (pi/180 * Rearth);
#   d_ADT   = sqrt(dx_ADT.^2 + dy_ADT.^2);

def compute_gradient_magnitude(adt_2d, lat_1d, grid_spacing_deg):
    """
    Compute |grad(ADT)| in m/m from a single daily ADT field.

    Parameters
    ----------
    adt_2d : np.ndarray, shape (nlat, nlon)
        Absolute dynamic topography in metres.
    lat_1d : np.ndarray, shape (nlat,)
        Latitude vector in degrees.
    grid_spacing_deg : float
        Grid spacing in degrees (0.125 for DT-2024, 0.25 for DT-2021).

    Returns
    -------
    grad_mag : np.ndarray, shape (nlat, nlon)
        Gradient magnitude in m/m.
    """
    # MATLAB: [dx, dy] = gradient(ADT, grid_spacing_deg)
    # np.gradient returns gradient per array index → divide by grid spacing
    # axis=1 → d/dlon, axis=0 → d/dlat
    dy_raw, dx_raw = np.gradient(adt_2d, grid_spacing_deg)

    # Convert to physical units (m per m)
    # MATLAB: dx_ADT = dx / (pi/180 * R * cos(lat))
    cos_lat = np.cos(DEG2RAD * lat_1d)[:, np.newaxis]  # (nlat, 1)
    dx_phys = dx_raw / (DEG2RAD * EARTH_RADIUS * cos_lat)
    dy_phys = dy_raw / (DEG2RAD * EARTH_RADIUS)

    grad_mag = np.sqrt(dx_phys**2 + dy_phys**2)
    return grad_mag


# =============================================================================
# 2. HELPER: ZERO-FREQUENCY-CROSSING WIDTH
# =============================================================================
# The manuscript (Sec 2.2) defines width as:
#   "the total meridional distance between zero-frequency crossings
#    on either side of the dominant axis"
#
# This replaces MATLAB's findpeaks(...,'WidthReference','halfheight').

def zero_crossing_width(freq_profile, lat_1d, peak_idx):
    """
    Compute meander width as the meridional distance between the first
    zero-frequency crossings north and south of the peak.

    Parameters
    ----------
    freq_profile : np.ndarray, shape (nlat,)
        Meander occurrence frequency at a single longitude.
    lat_1d : np.ndarray, shape (nlat,)
        Latitude vector (south to north, i.e., increasing).
    peak_idx : int
        Index of the peak in freq_profile.

    Returns
    -------
    width_deg : float
        Width in degrees latitude.
    south_lat : float
        Latitude of southern zero crossing.
    north_lat : float
        Latitude of northern zero crossing.
    """
    n = len(freq_profile)

    # Search southward (decreasing index if lat is ascending)
    south_lat = lat_1d[0]  # default: domain edge
    for j in range(peak_idx - 1, -1, -1):
        if freq_profile[j] <= 0 or np.isnan(freq_profile[j]):
            south_lat = lat_1d[j]
            break

    # Search northward (increasing index if lat is ascending)
    north_lat = lat_1d[-1]  # default: domain edge
    for j in range(peak_idx + 1, n):
        if freq_profile[j] <= 0 or np.isnan(freq_profile[j]):
            north_lat = lat_1d[j]
            break

    width_deg = north_lat - south_lat
    return width_deg, south_lat, north_lat


# =============================================================================
# 3. MAIN: PROCESS ONE SITE
# =============================================================================
def process_site(site_key, adt_fp, out_dir,
                 relat_thresh=RELAT_THRESH, x_months=X_MONTHS):
    """
    Full meander detection pipeline for a single site.

    Faithfully adapted from f_derive_meander_location_daily.m.

    Steps (matching MATLAB line numbers):
      1. Load ADT for the site outer domain (line 40-64)
      2. Compute daily gradient magnitude (line 66-79)
      3. Apply 20% relative threshold per day (line 131-136)
      4. Build monthly occurrence maps over 4-month window (line 181-197)
      5. Extract meander centre, width, edges per month per longitude (line 226-268)
      6. Save results
    """
    site = SITES[site_key]
    print(f"\n{'='*70}")
    print(f"Processing: {site['name']} ({site_key})")
    print(f"  Outer domain: lon {site['outer_lon']}, lat {site['outer_lat']}")
    print(f"  Inner domain: lon {site['inner_lon']}, lat {site['inner_lat']}")
    print(f"  Threshold: {relat_thresh*100:.0f}%  |  Aggregation: {x_months} months")
    print(f"{'='*70}")

    t0 = _time.time()

    # ------------------------------------------------------------------
    # Step 1: Load data and identify coordinates
    # ------------------------------------------------------------------
    # Adapted from MATLAB lines 40-64
    print("Step 1: Loading ADT data...")

    with Dataset(str(adt_fp), "r") as src:
        lat_name = "latitude" if "latitude" in src.variables else "lat"
        lon_name = "longitude" if "longitude" in src.variables else "lon"

        lat_full = np.array(src.variables[lat_name][:], dtype=np.float64)
        lon_full = np.array(src.variables[lon_name][:], dtype=np.float64)
        time_raw = src.variables["time"]
        time_all = pd.DatetimeIndex(
            xr.coding.times.decode_cf_datetime(
                time_raw[:], time_raw.units,
                calendar=getattr(time_raw, "calendar", "standard"),
            )
        )

        # Detect grid spacing
        grid_spacing = round(abs(float(lat_full[1] - lat_full[0])), 4)
        print(f"  Grid spacing detected: {grid_spacing}°")
        print(f"  Time range: {time_all[0].date()} to {time_all[-1].date()} ({len(time_all)} days)")
        print(f"  File longitude range: {lon_full[0]:.2f} to {lon_full[-1]:.2f}")

        # Outer domain indices (for gradient computation)
        olon = site["outer_lon"]
        olat = site["outer_lat"]
        wraps = site.get("wraps", False)

        ilat_outer = np.where((lat_full >= olat[0]) & (lat_full <= olat[1]))[0]
        ilon_outer = select_lon_indices(lon_full, olon[0], olon[1], wraps)

        assert len(ilat_outer) > 0, f"No lat points found in {olat}"
        assert len(ilon_outer) > 0, f"No lon points found in {olon} (wraps={wraps})"

        lat_outer = lat_full[ilat_outer]
        lon_outer = lon_full[ilon_outer]

        # For non-wrapping: indices are contiguous → single slice
        # For wrapping (dateline): indices have a gap → read two slices and concatenate
        if not wraps:
            lon_slice = slice(int(ilon_outer[0]), int(ilon_outer[-1]) + 1)
            lon_contiguous = True
        else:
            # In a -180..180 file with wraps=True:
            #   low indices  = western side (-180 to -150)
            #   high indices = eastern side (150 to 180)
            # Geographic order: east first (150→180), then west (-180→-150)
            gap = np.where(np.diff(ilon_outer) > 1)[0]
            if len(gap) > 0:
                split = gap[0] + 1
                ilon_west = ilon_outer[:split]     # low indices: -180 to -150
                ilon_east = ilon_outer[split:]      # high indices: 150 to 180
                lon_slice_east = slice(int(ilon_east[0]), int(ilon_east[-1]) + 1)
                lon_slice_west = slice(int(ilon_west[0]), int(ilon_west[-1]) + 1)
                lon_contiguous = False
                # Reorder to geographic continuity: 150→180, -180→-150
                ilon_outer = np.concatenate([ilon_east, ilon_west])
                lon_outer = lon_full[ilon_outer]
                print(f"  Dateline crossing: reading lon in two parts "
                      f"[{lon_full[ilon_east[0]]:.1f}..{lon_full[ilon_east[-1]]:.1f}] + "
                      f"[{lon_full[ilon_west[0]]:.1f}..{lon_full[ilon_west[-1]]:.1f}]")
            else:
                lon_slice = slice(int(ilon_outer[0]), int(ilon_outer[-1]) + 1)
                lon_contiguous = True

        lat_slice = slice(int(ilat_outer[0]), int(ilat_outer[-1]) + 1)

        print(f"  Outer domain: {len(lat_outer)} lat × {len(lon_outer)} lon points")

        # Inner domain indices (relative to outer domain arrays)
        ilon_inner = site["inner_lon"]
        ilat_inner = site["inner_lat"]
        inner_lat_mask = (lat_outer >= ilat_inner[0]) & (lat_outer <= ilat_inner[1])
        inner_lon_mask = select_lon_indices(lon_outer, ilon_inner[0], ilon_inner[1], wraps).astype(bool)
        # Convert index array to boolean mask over lon_outer
        _inner_lon_bool = np.zeros(len(lon_outer), dtype=bool)
        _inner_lon_idx = select_lon_indices(lon_outer, ilon_inner[0], ilon_inner[1], wraps)
        _inner_lon_bool[_inner_lon_idx] = True
        inner_lon_mask = _inner_lon_bool

        lat_inner = lat_outer[inner_lat_mask]
        lon_inner = lon_outer[inner_lon_mask]
        inner_lat_idx = np.where(inner_lat_mask)[0]
        inner_lon_idx = np.where(inner_lon_mask)[0]

        print(f"  Inner domain: {len(lat_inner)} lat × {len(lon_inner)} lon points")

        nt = len(time_all)

        # ------------------------------------------------------------------
        # Step 2 & 3: Compute gradients and apply threshold (chunked)
        # ------------------------------------------------------------------
        # Adapted from MATLAB lines 66-79 (gradient) and 131-136 (threshold)
        print("Step 2–3: Computing daily gradients and applying threshold...")

        # Pre-allocate daily front mask (1=front, 0=no front)
        # Store as uint8 to save memory
        front_mask = np.zeros((nt, len(lat_outer), len(lon_outer)), dtype=np.uint8)

        TIME_CHUNK = 30  # days per chunk
        for chunk_start in range(0, nt, TIME_CHUNK):
            chunk_end = min(chunk_start + TIME_CHUNK, nt)
            # Read ADT chunk: shape (chunk_size, nlat, nlon)
            if lon_contiguous:
                adt_chunk = np.array(
                    src.variables["adt"][chunk_start:chunk_end, lat_slice, lon_slice],
                    dtype=np.float32,
                )
            else:
                # Dateline crossing: read two parts and concatenate along lon axis
                east = np.array(
                    src.variables["adt"][chunk_start:chunk_end, lat_slice, lon_slice_east],
                    dtype=np.float32,
                )
                west = np.array(
                    src.variables["adt"][chunk_start:chunk_end, lat_slice, lon_slice_west],
                    dtype=np.float32,
                )
                adt_chunk = np.concatenate([east, west], axis=2)

            for k in range(adt_chunk.shape[0]):
                day_idx = chunk_start + k
                adt_day = adt_chunk[k]

                # Replace fill values with NaN
                adt_day = np.where(np.abs(adt_day) > 100, np.nan, adt_day)

                # Compute gradient magnitude (MATLAB lines 70-78)
                grad_mag = compute_gradient_magnitude(adt_day, lat_outer, grid_spacing)

                # Apply relative threshold (MATLAB lines 133-136)
                # thresh_val = max(max(d_ADT)) * relat_thresh
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    thresh_val = np.nanmax(grad_mag) * relat_thresh

                if np.isfinite(thresh_val) and thresh_val > 0:
                    front_mask[day_idx] = (grad_mag > thresh_val).astype(np.uint8)

            if (chunk_start // TIME_CHUNK) % 50 == 0:
                elapsed = _time.time() - t0
                print(f"    Processed days {chunk_start}–{chunk_end-1} of {nt} "
                      f"({elapsed:.0f}s elapsed)")

    print(f"  Front mask computed: shape {front_mask.shape}")

    # ------------------------------------------------------------------
    # Step 4: Build monthly occurrence maps
    # ------------------------------------------------------------------
    # Adapted from MATLAB lines 181-197
    print("Step 4: Building monthly occurrence maps...")

    # Define monthly timestamps (15th of each month) — MATLAB line 183
    first_month = time_all[0].to_period("M").to_timestamp() + pd.Timedelta(days=14)
    last_month = time_all[-1].to_period("M").to_timestamp()
    monthly_dates = pd.date_range(first_month, last_month, freq="MS") + pd.Timedelta(days=14)

    # Find index of each monthly date in daily time — MATLAB line 187
    idx_monthly = time_all.get_indexer(monthly_dates, method="nearest")

    # Convert x_months to x_days — MATLAB line 191
    x_days = int(2 * np.floor(x_months * 30.417 / 2))
    half = x_days // 2

    n_months = len(monthly_dates)
    n_lat_inner = len(lat_inner)
    n_lon_inner = len(lon_inner)

    # Pre-allocate occurrence maps for the inner domain only
    # MATLAB line 194-197: running sum of front_mask over ±half days
    occurrence = np.zeros((n_months, n_lat_inner, n_lon_inner), dtype=np.int16)

    # Valid month range (MATLAB: floor(x_months/2+1) to length-floor(x_months/2))
    valid_start = int(np.floor(x_months / 2))
    valid_end = n_months - int(np.floor(x_months / 2))

    for mi in range(valid_start, valid_end):
        center_day_idx = idx_monthly[mi]
        s = center_day_idx - (half - 1)
        e = center_day_idx + half + 1  # Python slicing is exclusive at end

        if s < 0 or e > nt:
            continue

        # Sum daily masks within the window, subsetting to inner domain
        window = front_mask[s:e, :, :][:, inner_lat_idx[0]:inner_lat_idx[-1]+1,
                                         inner_lon_idx[0]:inner_lon_idx[-1]+1]
        occurrence[mi] = window.sum(axis=0)

    print(f"  Occurrence maps: shape {occurrence.shape}, valid months {valid_start}–{valid_end-1}")

    # Free the large front_mask array
    del front_mask
    gc.collect()

    # ------------------------------------------------------------------
    # Step 5: Extract meander centre, width, edges
    # ------------------------------------------------------------------
    # Adapted from MATLAB lines 226-268
    print("Step 5: Extracting meander centre, width, and edges...")

    # Output arrays
    center_lat = np.full((n_months, n_lon_inner), np.nan)
    peak_freq  = np.full((n_months, n_lon_inner), np.nan)
    width_deg  = np.full((n_months, n_lon_inner), np.nan)
    south_edge = np.full((n_months, n_lon_inner), np.nan)
    north_edge = np.full((n_months, n_lon_inner), np.nan)

    for mi in range(valid_start, valid_end):
        for li in range(n_lon_inner):
            # Meridional frequency profile at this longitude — MATLAB line 248
            freq_profile = occurrence[mi, :, li].astype(np.float64)

            # Skip if all zero
            if freq_profile.max() <= 0:
                continue

            # Find peaks — MATLAB: [pks,locs,widths,proms] = findpeaks(...)
            peaks_idx, properties = find_peaks(freq_profile, height=1)

            if len(peaks_idx) == 0:
                continue

            # Find the biggest peak — MATLAB line 250: [maxf, idx] = max(pks)
            peak_heights = properties["peak_heights"]
            best = np.argmax(peak_heights)
            best_idx = peaks_idx[best]

            center_lat[mi, li] = lat_inner[best_idx]
            peak_freq[mi, li]  = peak_heights[best]

            # Width via zero-frequency crossing (manuscript definition)
            w, s_lat, n_lat = zero_crossing_width(freq_profile, lat_inner, best_idx)
            width_deg[mi, li]  = w
            south_edge[mi, li] = s_lat
            north_edge[mi, li] = n_lat

    print(f"  Detection complete. Valid data fraction: "
          f"{np.isfinite(center_lat).sum() / center_lat.size:.1%}")

    # ------------------------------------------------------------------
    # Step 6: Compute site-averaged monthly time series
    # ------------------------------------------------------------------
    print("Step 6: Computing site-averaged monthly metrics...")

    # Average across longitudes for each month (ignoring NaN)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_center_lat = np.nanmean(center_lat, axis=1)
        mean_width_deg  = np.nanmean(width_deg, axis=1)
        mean_peak_freq  = np.nanmean(peak_freq, axis=1)

    # Convert width from degrees to km
    mean_width_km = mean_width_deg * 111.32

    # Build DataFrame
    df = pd.DataFrame({
        "date": monthly_dates,
        "center_lat": mean_center_lat,
        "width_deg": mean_width_deg,
        "width_km": mean_width_km,
        "peak_frequency": mean_peak_freq,
    })
    df = df.set_index("date")

    # Mark invalid months as NaN
    df.iloc[:valid_start] = np.nan
    df.iloc[valid_end:] = np.nan

    # ------------------------------------------------------------------
    # Step 7: Save results
    # ------------------------------------------------------------------
    print("Step 7: Saving results...")

    # Save NetCDF with full spatial fields
    out_nc = out_dir / f"meander_detection_{site_key}_rel{int(relat_thresh*100)}_x{x_months}m.nc"
    ds_out = xr.Dataset(
        {
            "occurrence":   (["month", "latitude", "longitude"], occurrence),
            "center_lat":   (["month", "longitude"], center_lat),
            "peak_freq":    (["month", "longitude"], peak_freq),
            "width_deg":    (["month", "longitude"], width_deg),
            "south_edge":   (["month", "longitude"], south_edge),
            "north_edge":   (["month", "longitude"], north_edge),
        },
        coords={
            "month":     monthly_dates,
            "latitude":  lat_inner,
            "longitude": lon_inner,
        },
        attrs={
            "site": site["name"],
            "site_key": site_key,
            "relat_thresh": relat_thresh,
            "x_months": x_months,
            "x_days": x_days,
            "grid_spacing_deg": grid_spacing,
            "source_file": str(adt_fp),
            "width_definition": "zero-frequency crossing (Liu et al. 2024, 2026)",
            "threshold_definition": "20% of local outer-domain maximum per day",
        },
    )
    ds_out.to_netcdf(out_nc)
    print(f"  Saved: {out_nc}")

    # Save CSV time series
    out_csv = out_dir / f"monthly_metrics_{site_key}.csv"
    df.to_csv(out_csv)
    print(f"  Saved: {out_csv}")

    elapsed = _time.time() - t0
    print(f"\n  {site_key} complete in {elapsed/60:.1f} minutes.")

    return df, ds_out


# =============================================================================
# 4. MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("GRL Circumpolar Meander Detection Pipeline")
    print(f"  Source: {ADT_FP}")
    print(f"  Threshold: {RELAT_THRESH*100:.0f}% | Aggregation: {X_MONTHS} months")
    print(f"  Sites: {list(SITES.keys())}")
    print("=" * 70)

    # Check input file exists
    if not ADT_FP.exists():
        raise FileNotFoundError(
            f"ADT file not found: {ADT_FP}\n"
            "Please update ADT_FP in the CONFIGURATION section."
        )

    all_results = {}
    for site_key in SITES:
        df, ds = process_site(site_key, ADT_FP, OUT_DIR)
        all_results[site_key] = df
        del ds
        gc.collect()

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY: Monthly metric means (full record)")
    print("=" * 70)
    for key, df in all_results.items():
        print(f"\n{SITES[key]['name']} ({key}):")
        print(f"  Mean centre latitude: {df['center_lat'].mean():.2f}°S")
        print(f"  Mean width: {df['width_km'].mean():.1f} km")
        print(f"  Mean peak frequency: {df['peak_frequency'].mean():.1f} days")

    print("\n\nAll sites processed. Output in:", OUT_DIR)
