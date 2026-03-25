#!/usr/bin/env python3
"""
combine_cp_bathymetry.py
========================
Combine the two Campbell Plateau GMRT bathymetry files across the dateline
into a single continuous NetCDF file.

  CP_East: 150E to 180E  (kept as-is)
  CP_West: -180 to -150  (shifted to 180-210)
  Combined: ~150E to ~210E (continuous across dateline)

Run: python combine_cp_bathymetry.py
"""

import numpy as np
import xarray as xr
from pathlib import Path

GMRT_DIR = Path("/g/data/gv90/xl1657/gmrt")
FP_EAST  = GMRT_DIR / "GMRTv4_4_1_20260314topo_CP_East.grd"
FP_WEST  = GMRT_DIR / "GMRTv4_4_1_20260314topo_CP_West.grd"
FP_OUT   = GMRT_DIR / "GMRTv4_4_1_20260314topo_CP_Combined.nc"


def load_gmrt_raw(fp):
    """Load a GMT-format .grd file at full resolution."""
    ds = xr.open_dataset(fp)
    nx, ny = int(ds["dimension"].values[0]), int(ds["dimension"].values[1])
    x0, x1 = ds["x_range"].values
    y0, y1 = ds["y_range"].values
    lon = np.linspace(x0, x1, nx)
    lat = np.linspace(y0, y1, ny)
    z = ds["z"].values.reshape(ny, nx)
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        z = z[::-1, :]
    ds.close()
    return lon, lat, z


if __name__ == "__main__":
    print("=" * 50)
    print("Combining CP bathymetry across dateline")
    print("=" * 50)

    # Load both halves
    print(f"  Loading East: {FP_EAST.name}")
    lon_e, lat_e, z_e = load_gmrt_raw(FP_EAST)
    print(f"    lon: {lon_e[0]:.4f} to {lon_e[-1]:.4f}, shape: {z_e.shape}")

    print(f"  Loading West: {FP_WEST.name}")
    lon_w, lat_w, z_w = load_gmrt_raw(FP_WEST)
    print(f"    lon: {lon_w[0]:.4f} to {lon_w[-1]:.4f}, shape: {z_w.shape}")

    # Shift western lons: -180..-150 -> 180..210
    lon_w_shifted = lon_w + 360.0
    print(f"    lon shifted: {lon_w_shifted[0]:.4f} to {lon_w_shifted[-1]:.4f}")

    # Check latitude grids match
    if len(lat_e) == len(lat_w) and np.allclose(lat_e, lat_w, atol=0.001):
        print("  Latitude grids match — direct concatenation.")
        lat_combined = lat_e
        z_w_use = z_w
    else:
        print("  Latitude grids differ — interpolating west onto east grid.")
        from scipy.interpolate import RegularGridInterpolator
        interp = RegularGridInterpolator(
            (lat_w, lon_w_shifted), z_w,
            bounds_error=False, fill_value=np.nan, method="linear"
        )
        LAT_q, LON_q = np.meshgrid(lat_e, lon_w_shifted, indexing="ij")
        z_w_use = interp((LAT_q, LON_q))
        lat_combined = lat_e

    # Concatenate along longitude
    lon_combined = np.concatenate([lon_e, lon_w_shifted])
    z_combined = np.concatenate([z_e, z_w_use], axis=1)

    # Sort by longitude
    sort_idx = np.argsort(lon_combined)
    lon_combined = lon_combined[sort_idx]
    z_combined = z_combined[:, sort_idx]

    print(f"\n  Combined grid:")
    print(f"    lon: {lon_combined[0]:.4f} to {lon_combined[-1]:.4f} ({len(lon_combined)} points)")
    print(f"    lat: {lat_combined[0]:.4f} to {lat_combined[-1]:.4f} ({len(lat_combined)} points)")
    print(f"    z shape: {z_combined.shape}")
    print(f"    depth range: {np.nanmin(z_combined):.0f} to {np.nanmax(z_combined):.0f} m")

    # Save as standard CF-compliant NetCDF
    ds_out = xr.Dataset(
        {"elevation": (["latitude", "longitude"], z_combined)},
        coords={
            "latitude": lat_combined,
            "longitude": lon_combined,
        },
        attrs={
            "title": "GMRT Bathymetry — Campbell Plateau (combined across dateline)",
            "source": f"Combined from {FP_EAST.name} and {FP_WEST.name}",
            "convention": "Longitude in 0-360 degrees East",
            "history": "East (150-180E) + West (-180 to -150 shifted to 180-210)",
        },
    )
    ds_out["elevation"].attrs = {
        "units": "meters",
        "long_name": "Seafloor elevation relative to sea level",
        "positive": "up",
    }
    ds_out["latitude"].attrs = {"units": "degrees_north", "long_name": "Latitude"}
    ds_out["longitude"].attrs = {"units": "degrees_east", "long_name": "Longitude (0-360)"}

    ds_out.to_netcdf(FP_OUT)
    size_mb = FP_OUT.stat().st_size / 1024**2
    print(f"\n  Saved: {FP_OUT}")
    print(f"  Size: {size_mb:.1f} MB")
    print("Done.")
