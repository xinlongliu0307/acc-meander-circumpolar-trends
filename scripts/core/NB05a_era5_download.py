#!/usr/bin/env python3
"""
NB05a_era5_download.py
======================
Download ERA5 monthly-mean 10m u and v wind for the Southern Ocean
from the Copernicus Climate Data Store (CDS).

Downloads a single file covering all four sites:
  - Variables: 10m u-component, 10m v-component
  - Area: 30S to 70S, all longitudes (covers all 4 sites)
  - Period: 1993–2025
  - Product: monthly averaged reanalysis

Requires: pip install cdsapi
          ~/.cdsapirc with your CDS API credentials

NCI Gadi setup:
    source /g/data/gv90/xl1657/venvs/cmems_py311/bin/activate
"""

import cdsapi
from pathlib import Path

OUT_DIR = Path("/g/data/gv90/xl1657/era5_wind")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "era5_10m_wind_monthly_SO_1993_2025.nc"

if OUT_FILE.exists():
    print(f"Output file already exists: {OUT_FILE}")
    print("Delete it first if you want to re-download.")
    exit(0)

print("=" * 60)
print("ERA5 10m Wind Download")
print(f"  Output: {OUT_FILE}")
print("=" * 60)

# All years and months
years = [str(y) for y in range(1993, 2026)]
months = [f"{m:02d}" for m in range(1, 13)]

print(f"  Years: {years[0]}–{years[-1]}")
print(f"  Area: 30S–70S, all longitudes")
print(f"  Variables: 10m u-component, 10m v-component")
print(f"\nSubmitting request to CDS (this may take 10–30 minutes)...")

client = cdsapi.Client()

client.retrieve(
    "reanalysis-era5-single-levels-monthly-means",
    {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
        ],
        "year": years,
        "month": months,
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [-30, -180, -70, 180],  # [North, West, South, East]
    },
    str(OUT_FILE),
)

print(f"\nDownload complete: {OUT_FILE}")
print(f"File size: {OUT_FILE.stat().st_size / 1024**2:.1f} MB")
