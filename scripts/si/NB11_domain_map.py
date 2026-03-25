#!/usr/bin/env python3
"""
NB11_domain_map.py
==================
Step 3 of GRL Supporting Information: Domain Definition Map (Figure S3).

Produces a 4-panel figure showing each study site with:
  - GMRT bathymetry (shaded + contours)
  - Outer domain (dashed rectangle)
  - Inner domain (solid rectangle)
  - Time-mean meander centerline (from detection NetCDF)

Run on Gadi (login node is fine — no heavy computation):
  source /g/data/gv90/xl1657/venvs/cmems_py311/bin/activate
  module purge
  unset PYTHONPATH
  cd /g/data/gv90/xl1657/cmems_adt/notebooks/
  python NB11_domain_map.py

Author: Xinlong Liu, IMAS, University of Tasmania
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.colors import BoundaryNorm
import warnings

warnings.filterwarnings("ignore")

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("WARNING: Cartopy not found. Using plain matplotlib (no map projection).")

from pathlib import Path

# =============================================================================
# 0. CONFIGURATION
# =============================================================================

BASE_DIR = Path("/g/data/gv90/xl1657/cmems_adt")
PROD_DIR = BASE_DIR / "grl_meander_products"
GMRT_DIR = Path("/g/data/gv90/xl1657/gmrt")
FIG_DIR  = PROD_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Site definitions — coordinates for plotting
# outer_lon/lat: gradient computation domain (dashed box)
# inner_lon/lat: meander tracking domain (solid box)
# plot_lon/lat: map extent for the panel (slightly larger than outer)
SITES = {
    "CP": {
        "name": "Campbell Plateau",
        "outer_lon": (150, 210), "outer_lat": (-70, -30),
        "inner_lon": (150, 210), "inner_lat": (-57, -46),
        "plot_lon":  (145, 215), "plot_lat":  (-65, -35),
        "gmrt_file": GMRT_DIR / "GMRTv4_4_1_20260314topo_CP_Combined.nc",
        "gmrt_format": "nc",
        "color": "#0072B2",
    },
    "PAR": {
        "name": "Pacific-Antarctic Ridge",
        "outer_lon": (210, 300), "outer_lat": (-70, -30),
        "inner_lon": (210, 280), "inner_lat": (-60, -48),
        "plot_lon":  (205, 305), "plot_lat":  (-65, -35),
        "gmrt_file": GMRT_DIR / "GMRTv4_4_1_20260314topo_PAR.grd",
        "gmrt_format": "grd",
        "color": "#E69F00",
    },
    "SEIR": {
        "name": "Southeast Indian Ridge",
        "outer_lon": (120, 155), "outer_lat": (-65, -30),
        "inner_lon": (130, 152), "inner_lat": (-56, -44),
        "plot_lon":  (115, 160), "plot_lat":  (-62, -35),
        "gmrt_file": GMRT_DIR / "GMRTv4_4_1_20260314topo_SEIR.grd",
        "gmrt_format": "grd",
        "color": "#CC79A7",
    },
    "SWIR": {
        "name": "Southwest Indian Ridge",
        "outer_lon": (5, 55),   "outer_lat": (-65, -35),
        "inner_lon": (15, 45),  "inner_lat": (-58, -44),
        "plot_lon":  (0, 60),   "plot_lat":  (-62, -38),
        "gmrt_file": GMRT_DIR / "GMRTv4_4_1_20260314topo_SWIR.grd",
        "gmrt_format": "grd",
        "color": "#009E73",
    },
}

# Bathymetry contour levels (metres)
BATHY_LEVELS = [-6000, -5000, -4000, -3000, -2000, -1000, 0]
BATHY_CMAP_LEVELS = np.arange(-7000, 500, 500)


# =============================================================================
# 1. GMRT READERS
# =============================================================================

def read_gmrt_grd(filepath):
    """
    Read a GMT-format .grd file (GMRT).
    The z variable is a flat 1D array that must be reshaped
    using dimension, x_range, y_range metadata.
    """
    ds = xr.open_dataset(filepath)

    x_range = ds["x_range"].values      # [lon_min, lon_max]
    y_range = ds["y_range"].values      # [lat_min, lat_max]
    spacing = ds["spacing"].values      # [dlon, dlat]
    dimension = ds["dimension"].values  # [nlon, nlat]
    z = ds["z"].values                  # flat 1D

    nlon = int(dimension[0])
    nlat = int(dimension[1])

    lon = np.linspace(x_range[0], x_range[1], nlon)
    lat = np.linspace(y_range[0], y_range[1], nlat)

    # Reshape z: GMT stores row-major, top-to-bottom
    topo = z.reshape(nlat, nlon)

    # GMT convention: first row = northernmost latitude
    # Check if lat is ascending; if descending, flip
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        topo = topo[::-1, :]

    ds.close()
    return lon, lat, topo


def read_gmrt_nc(filepath):
    """
    Read the CP Combined NetCDF (standard format).
    Variable name could be 'altitude', 'z', or 'elevation'.
    """
    ds = xr.open_dataset(filepath)

    # Find the data variable
    for varname in ["altitude", "z", "elevation", "topo"]:
        if varname in ds:
            topo = ds[varname].values
            break
    else:
        # Use the first non-coordinate variable
        data_vars = [v for v in ds.data_vars if v not in
                     ["x_range", "y_range", "z_range", "spacing", "dimension"]]
        if data_vars:
            topo = ds[data_vars[0]].values
        else:
            raise ValueError(f"Cannot find topo variable in {filepath}")

    # Find coordinate names
    for xname in ["x", "lon", "longitude"]:
        if xname in ds.coords or xname in ds.dims:
            lon = ds[xname].values
            break
    else:
        # Try dimension-based
        dims = list(ds.dims.keys())
        lon = ds[dims[-1]].values if len(dims) >= 2 else np.arange(topo.shape[-1])

    for yname in ["y", "lat", "latitude"]:
        if yname in ds.coords or yname in ds.dims:
            lat = ds[yname].values
            break
    else:
        dims = list(ds.dims.keys())
        lat = ds[dims[-2]].values if len(dims) >= 2 else np.arange(topo.shape[-2])

    # If z is 1D (GMT format inside .nc), reshape
    if topo.ndim == 1:
        if "dimension" in ds:
            dim = ds["dimension"].values
            nlon, nlat = int(dim[0]), int(dim[1])
            topo = topo.reshape(nlat, nlon)
            x_range = ds["x_range"].values
            y_range = ds["y_range"].values
            lon = np.linspace(x_range[0], x_range[1], nlon)
            lat = np.linspace(y_range[0], y_range[1], nlat)
            if lat[0] > lat[-1]:
                lat = lat[::-1]
                topo = topo[::-1, :]
        else:
            raise ValueError(f"1D z but no dimension metadata in {filepath}")

    ds.close()
    return lon, lat, topo


def load_bathymetry(site_key):
    """Load GMRT bathymetry for a site."""
    site = SITES[site_key]
    fp = site["gmrt_file"]
    fmt = site["gmrt_format"]

    if not fp.exists():
        print(f"  WARNING: Bathymetry file not found: {fp}")
        return None, None, None

    print(f"  Loading bathymetry for {site_key}: {fp.name}")
    if fmt == "grd":
        return read_gmrt_grd(fp)
    else:
        return read_gmrt_nc(fp)


# =============================================================================
# 2. LOAD MEAN MEANDER CENTERLINE
# =============================================================================

def load_mean_centerline(site_key):
    """
    Load the time-mean meander centerline from the detection NetCDF.
    Returns (lon_inner, mean_center_lat) arrays.
    """
    nc_fp = PROD_DIR / f"meander_detection_{site_key}_rel20_x4m.nc"
    if not nc_fp.exists():
        print(f"  WARNING: Detection file not found: {nc_fp}")
        return None, None

    ds = xr.open_dataset(nc_fp)
    center_lat = ds["center_lat"].values   # (month, lon)
    lon_inner = ds["longitude"].values

    # Time-mean centerline
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_lat = np.nanmean(center_lat, axis=0)

    ds.close()
    return lon_inner, mean_lat


# =============================================================================
# 3. CONVERT LONGITUDES FOR PLOTTING
# =============================================================================

def to_180(lon):
    """Convert 0–360 longitude to -180–180."""
    return ((lon + 180) % 360) - 180

def to_360(lon):
    """Convert -180–180 longitude to 0–360."""
    return lon % 360


# =============================================================================
# 4. PLOTTING
# =============================================================================

def draw_domain_box(ax, lon_range, lat_range, color, linestyle, linewidth,
                    transform, label=None):
    """Draw a rectangular domain box on the map."""
    lon1, lon2 = lon_range
    lat1, lat2 = lat_range

    # For non-Cartopy axes or simple cases
    lons = [lon1, lon2, lon2, lon1, lon1]
    lats = [lat1, lat1, lat2, lat2, lat1]
    ax.plot(lons, lats, color=color, linestyle=linestyle,
            linewidth=linewidth, transform=transform, label=label, zorder=5)


def plot_site_panel(ax, site_key, panel_label):
    """Plot one site panel with bathymetry, domains, and centerline."""
    site = SITES[site_key]
    color = site["color"]

    # Load bathymetry
    blon, blat, topo = load_bathymetry(site_key)

    # Determine plot extent and projection
    plon = site["plot_lon"]
    plat = site["plot_lat"]

    # Convert plot extent to -180/180 for Cartopy
    plon_180 = (to_180(plon[0]), to_180(plon[1]))

    # Handle dateline-crossing sites (CP)
    crosses_dateline = plon_180[0] > plon_180[1]

    if HAS_CARTOPY:
        if crosses_dateline:
            central_lon = 180
        else:
            central_lon = (plon_180[0] + plon_180[1]) / 2

        proj = ccrs.PlateCarree(central_longitude=central_lon)
        transform = ccrs.PlateCarree()

        ax.set_global()

        if crosses_dateline:
            ax.set_extent([plon_180[0], plon_180[1] + 360, plat[0], plat[1]],
                          crs=ccrs.PlateCarree(central_longitude=0))
        else:
            ax.set_extent([plon_180[0], plon_180[1], plat[0], plat[1]],
                          crs=transform)
    else:
        transform = ax.transData
        ax.set_xlim(plon_180)
        ax.set_ylim(plat)

    # Plot bathymetry
    if blon is not None:
        # Clip topo to sea only
        topo_sea = np.where(topo <= 0, topo, np.nan)

        # Subsample for performance if very large
        max_pts = 2000
        if len(blon) > max_pts:
            step_lon = max(1, len(blon) // max_pts)
            step_lat = max(1, len(blat) // max_pts)
            blon_sub = blon[::step_lon]
            blat_sub = blat[::step_lat]
            topo_sub = topo_sea[::step_lat, ::step_lon]
        else:
            blon_sub, blat_sub, topo_sub = blon, blat, topo_sea

        # Create coloured bathymetry
        levels = np.arange(-7000, 500, 500)
        norm = BoundaryNorm(levels, ncolors=256, clip=True)

        if HAS_CARTOPY:
            cf = ax.contourf(blon_sub, blat_sub, topo_sub, levels=levels,
                             cmap="bone", norm=norm, transform=transform,
                             zorder=1, extend="both")
            # Contour lines at key depths
            ax.contour(blon_sub, blat_sub, topo_sub,
                       levels=[-5000, -4000, -3000, -2000, -1000],
                       colors="grey", linewidths=0.3, transform=transform,
                       zorder=2)
            # Zero contour (coastline)
            ax.contour(blon_sub, blat_sub, topo_sub, levels=[0],
                       colors="black", linewidths=0.6, transform=transform,
                       zorder=3)
        else:
            cf = ax.contourf(blon_sub, blat_sub, topo_sub, levels=levels,
                             cmap="bone", norm=norm, zorder=1, extend="both")
            ax.contour(blon_sub, blat_sub, topo_sub,
                       levels=[-5000, -4000, -3000, -2000, -1000],
                       colors="grey", linewidths=0.3, zorder=2)

    # Add land
    if HAS_CARTOPY:
        ax.add_feature(cfeature.LAND, facecolor="black", zorder=4)

    # Draw outer domain box (dashed)
    olon = site["outer_lon"]
    olat = site["outer_lat"]
    olon_180 = (to_180(olon[0]), to_180(olon[1]))
    if olon_180[0] > olon_180[1]:
        # Dateline crossing: draw two segments
        draw_domain_box(ax, (olon_180[0], 180), olat, color, "--", 1.5, transform)
        draw_domain_box(ax, (-180, olon_180[1]), olat, color, "--", 1.5, transform)
    else:
        draw_domain_box(ax, olon_180, olat, color, "--", 1.5, transform,
                        label="Outer domain")

    # Draw inner domain box (solid)
    ilon = site["inner_lon"]
    ilat = site["inner_lat"]
    ilon_180 = (to_180(ilon[0]), to_180(ilon[1]))
    if ilon_180[0] > ilon_180[1]:
        draw_domain_box(ax, (ilon_180[0], 180), ilat, color, "-", 2.0, transform)
        draw_domain_box(ax, (-180, ilon_180[1]), ilat, color, "-", 2.0, transform)
    else:
        draw_domain_box(ax, ilon_180, ilat, color, "-", 2.0, transform,
                        label="Inner domain")

    # Plot mean meander centerline
    clon, clat = load_mean_centerline(site_key)
    if clon is not None:
        clon_180 = to_180(clon)
        valid = np.isfinite(clat)
        if HAS_CARTOPY:
            ax.plot(clon_180[valid], clat[valid], color=color,
                    linewidth=2.0, transform=transform, zorder=6,
                    label="Mean centerline")
        else:
            ax.plot(clon_180[valid], clat[valid], color=color,
                    linewidth=2.0, zorder=6, label="Mean centerline")

    # Gridlines
    if HAS_CARTOPY:
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="grey",
                          alpha=0.5, linestyle="--", zorder=7)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 7}
        gl.ylabel_style = {"size": 7}
    else:
        ax.grid(True, linewidth=0.3, color="grey", alpha=0.5, linestyle="--")
        ax.tick_params(labelsize=7)

    # Panel label and title
    ax.set_title(f"{site['name']} ({site_key})", fontsize=9, fontweight="bold",
                 pad=6)
    ax.text(0.03, 0.95, panel_label, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="none", alpha=0.8),
            zorder=10)


# =============================================================================
# 5. MAIN FIGURE
# =============================================================================

def generate_figure_s3():
    """Generate the 4-panel domain definition map."""
    print("=" * 60)
    print("Generating Figure S3: Domain Definition Map")
    print("=" * 60)

    site_keys = ["CP", "PAR", "SEIR", "SWIR"]
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    if HAS_CARTOPY:
        fig = plt.figure(figsize=(10, 10))

        # Create subplots with individual projections
        axes = []
        for i, site_key in enumerate(site_keys):
            site = SITES[site_key]
            plon = site["plot_lon"]
            plon_180 = (to_180(plon[0]), to_180(plon[1]))

            if plon_180[0] > plon_180[1]:
                central_lon = 180
            else:
                central_lon = (plon_180[0] + plon_180[1]) / 2

            proj = ccrs.PlateCarree(central_longitude=central_lon)
            ax = fig.add_subplot(2, 2, i + 1, projection=proj)
            axes.append(ax)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes = axes.flatten()

    for i, (site_key, label) in enumerate(zip(site_keys, panel_labels)):
        print(f"\nPlotting {site_key}...")
        plot_site_panel(axes[i], site_key, label)

    # Legend at bottom
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="grey", linewidth=1.5, linestyle="--",
               label="Outer domain (gradient computation)"),
        Line2D([0], [0], color="grey", linewidth=2.0, linestyle="-",
               label="Inner domain (meander tracking)"),
        Line2D([0], [0], color="grey", linewidth=2.0, linestyle="-",
               label="Time-mean meander centerline"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))

    plt.subplots_adjust(hspace=0.25, wspace=0.20)

    # Save
    out_pdf = FIG_DIR / "figure_s3_domain_map.pdf"
    out_png = FIG_DIR / "figure_s3_domain_map.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nFigure S3 saved: {out_pdf}")
    print(f"Figure S3 saved: {out_png}")


if __name__ == "__main__":
    generate_figure_s3()
    print("\n" + "=" * 60)
    print("DONE.")
    print("=" * 60)
