#!/usr/bin/env python3
"""
NB06_fig1.py -- Figure 1: Study Region Map
Run: python NB06_fig1.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import warnings

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

BASE_DIR = Path("/g/data/gv90/xl1657/cmems_adt/grl_meander_products")
GMRT_DIR = Path("/g/data/gv90/xl1657/gmrt")
FIG_DIR  = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CB = {"CP": "#0072B2", "PAR": "#E69F00", "SEIR": "#CC79A7", "SWIR": "#009E73"}

SITES = {
    "SWIR": {"name": "Southwest Indian Ridge",  "color": CB["SWIR"],
             "inner_lon": (15, 45),   "inner_lat": (-58, -44), "wraps": False,
             "zoom_lon": (5, 55),     "zoom_lat": (-62, -38)},
    "SEIR": {"name": "Southeast Indian Ridge",  "color": CB["SEIR"],
             "inner_lon": (130, 152), "inner_lat": (-56, -44), "wraps": False,
             "zoom_lon": (120, 160),  "zoom_lat": (-60, -38)},
    "CP":   {"name": "Campbell Plateau",         "color": CB["CP"],
             "inner_lon": (150, 210), "inner_lat": (-57, -46), "wraps": True,
             "zoom_lon": (148, 212),  "zoom_lat": (-60, -42)},
    "PAR":  {"name": "Pacific-Antarctic Ridge",  "color": CB["PAR"],
             "inner_lon": (210, 280), "inner_lat": (-60, -48), "wraps": False,
             "zoom_lon": (205, 290),  "zoom_lat": (-64, -42)},
}

GMRT_FILES = {
    "CP_east": GMRT_DIR / "GMRTv4_4_1_20260314topo_CP_East.grd",
    "CP_west": GMRT_DIR / "GMRTv4_4_1_20260314topo_CP_West.grd",
    "PAR":     GMRT_DIR / "GMRTv4_4_1_20260314topo_PAR.grd",
    "SEIR":    GMRT_DIR / "GMRTv4_4_1_20260314topo_SEIR.grd",
    "SWIR":    GMRT_DIR / "GMRTv4_4_1_20260314topo_SWIR.grd",
}

DECADES = [
    ("1993\u20132002", "1993-01", "2002-12", 0.8, "-"),
    ("2003\u20132014", "2003-01", "2014-12", 1.4, "--"),
    ("2015\u20132025", "2015-01", "2025-12", 2.2, "-"),
]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.linewidth": 0.8, "axes.labelsize": 9,
    "axes.titlesize": 10, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "legend.framealpha": 0.9, "figure.dpi": 150,
})


# ─── data loading ───
def load_all_data():
    print("Loading data...")
    data = {}
    for key in SITES:
        fp = BASE_DIR / f"monthly_metrics_{key}.csv"
        if fp.exists():
            data[key] = pd.read_csv(fp, index_col=0, parse_dates=True)
            print(f"  {key}: {len(data[key])} months")
    for key in SITES:
        fp = BASE_DIR / f"meander_detection_{key}_rel20_x4m.nc"
        if fp.exists():
            data[f"nc_{key}"] = xr.open_dataset(fp)
    return data


def load_gmrt(fp, coarsen=4):
    """Load a single GMRT .grd file."""
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
    if coarsen > 1:
        lon, lat, z = lon[::coarsen], lat[::coarsen], z[::coarsen, ::coarsen]
    return lon, lat, z


def load_gmrt_cp_combined(coarsen=4):
    """
    Load and combine the two CP GMRT files across the dateline.
    CP_East: 150E to 180E  -> keep as-is (150-180)
    CP_West: -180 to -150  -> shift to 180-210
    Result: continuous lon array from ~150 to ~210, shared lat grid.
    """
    fp_east = GMRT_FILES["CP_east"]
    fp_west = GMRT_FILES["CP_west"]

    lon_e, lat_e, z_e = load_gmrt(fp_east, coarsen=coarsen)
    lon_w, lat_w, z_w = load_gmrt(fp_west, coarsen=coarsen)

    # Shift western lons: -180..-150 -> 180..210
    lon_w_shifted = lon_w + 360.0

    # Interpolate west onto east's latitude grid if they differ slightly
    if len(lat_e) != len(lat_w) or not np.allclose(lat_e, lat_w, atol=0.01):
        from scipy.interpolate import RegularGridInterpolator
        interp = RegularGridInterpolator((lat_w, lon_w_shifted), z_w,
                                         bounds_error=False, fill_value=np.nan)
        LAT_q, LON_q = np.meshgrid(lat_e, lon_w_shifted, indexing="ij")
        z_w_regrid = interp((LAT_q, LON_q))
        lat_combined = lat_e
    else:
        z_w_regrid = z_w
        lat_combined = lat_e

    # Concatenate along longitude
    lon_combined = np.concatenate([lon_e, lon_w_shifted])
    z_combined = np.concatenate([z_e, z_w_regrid], axis=1)

    # Sort by longitude
    sort_idx = np.argsort(lon_combined)
    lon_combined = lon_combined[sort_idx]
    z_combined = z_combined[:, sort_idx]

    return lon_combined, lat_combined, z_combined


def load_bathy_for_site(site_key, coarsen_overview=10, coarsen_zoom=4, for_overview=False):
    """Load bathymetry for a site, combining CP halves."""
    c = coarsen_overview if for_overview else coarsen_zoom
    if site_key == "CP":
        return load_gmrt_cp_combined(coarsen=c)
    else:
        fp = GMRT_FILES.get(site_key)
        if fp and fp.exists():
            lon, lat, z = load_gmrt(fp, coarsen=c)
            # For PAR, shift negative lons to 0-360 for consistency
            if site_key == "PAR":
                lon = np.where(lon < 0, lon + 360, lon)
            return lon, lat, z
    return None, None, None


def draw_site_box(ax, site, proj, lw=2.5):
    """Draw inner domain rectangle."""
    lo0, lo1 = site["inner_lon"]
    la0, la1 = site["inner_lat"]
    col = site["color"]
    ax.plot([lo0, lo1, lo1, lo0, lo0],
            [la0, la0, la1, la1, la0],
            color=col, lw=lw, ls="-", transform=proj, zorder=5)


# ─── main figure ───
def plot_fig1(data):
    print("\nGenerating Figure 1...")
    if not HAS_CARTOPY:
        print("  Skipping (no Cartopy).")
        return

    # Use central_longitude=180 so 0-360 longitudes display correctly
    # This puts the Pacific at centre, matching the circumpolar ACC view
    proj = ccrs.PlateCarree(central_longitude=180)
    tr_data = ccrs.PlateCarree()  # data is always in standard lon

    fig = plt.figure(figsize=(14, 11))
    gs = gridspec.GridSpec(2, 4, figure=fig, height_ratios=[1, 1.1],
                           hspace=0.28, wspace=0.30)

    # =================================================================
    # Panel (a): Circumpolar overview
    # =================================================================
    ax_a = fig.add_subplot(gs[0, :], projection=proj)
    ax_a.set_global()
    ax_a.set_extent([0, 360, -75, -30], tr_data)
    ax_a.add_feature(cfeature.LAND, facecolor="black", edgecolor="black")
    ax_a.coastlines(resolution="50m", linewidth=0.3, color="0.3")

    gl = ax_a.gridlines(crs=tr_data, draw_labels=True, linewidth=0.2,
                        color="gray", alpha=0.3, linestyle="--")
    gl.xlocator = mticker.FixedLocator(np.arange(0, 361, 30))
    gl.ylocator = mticker.FixedLocator(np.arange(-70, -25, 10))
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}
    gl.top_labels = False
    gl.right_labels = False

    # Bathymetry overview (all sites, 0-360 convention)
    pcm_a = None
    for site_key in SITES:
        lon_b, lat_b, z_b = load_bathy_for_site(site_key, for_overview=True)
        if lon_b is not None:
            try:
                LON, LAT = np.meshgrid(lon_b, lat_b)
                pcm_a = ax_a.pcolormesh(LON, LAT, z_b, transform=tr_data,
                                       cmap="bone", vmin=-6000, vmax=0,
                                       shading="auto", alpha=0.70,
                                       rasterized=True)
            except Exception as e:
                print(f"    Warning overview {site_key}: {e}")

    # Site boxes on overview
    for sk, site in SITES.items():
        draw_site_box(ax_a, site, tr_data, lw=2.0)

    # Geographic labels (convert to 0-360 for plotting)
    for txt, lon360, lat, col in [
        ("South\nAmerica", 290, -38, "0.35"),
        ("Africa", 22, -33, "0.35"),
        ("Australia", 133, -33, "0.35"),
        ("New\nZealand", 174, -42, "0.35"),
        ("Southern Ocean", 180, -58, "0.45"),
        ("Antarctica", 180, -72, "0.55"),
    ]:
        ax_a.text(lon360, lat, txt, transform=tr_data, fontsize=7.5,
                 color=col, fontstyle="italic", ha="center", va="center")

    # Site name labels
    label_pos = {"SWIR": (30, -38), "SEIR": (141, -38),
                 "CP": (180, -40), "PAR": (245, -42)}
    for sk, site in SITES.items():
        lx, ly = label_pos[sk]
        ax_a.text(lx, ly, site["name"], transform=tr_data, fontsize=7,
                 fontweight="bold", color=site["color"], ha="center",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec=site["color"], alpha=0.9, lw=0.6))

    ax_a.text(0.02, 0.95, "(a)", transform=ax_a.transAxes,
             fontsize=13, fontweight="bold", va="top")

    if pcm_a is not None:
        cbar = fig.colorbar(pcm_a, ax=ax_a, orientation="vertical",
                           fraction=0.015, pad=0.02, shrink=0.75)
        cbar.set_label("Depth (m)", fontsize=8)
        cbar.ax.tick_params(labelsize=6)

    ax_a.set_ylabel("Latitude", fontsize=8)

    # =================================================================
    # Panels (b)-(e): Zoomed site views
    # =================================================================
    panel_labels = ["(b)", "(c)", "(d)", "(e)"]
    site_order = ["SWIR", "SEIR", "CP", "PAR"]

    for idx, site_key in enumerate(site_order):
        site = SITES[site_key]
        ax = fig.add_subplot(gs[1, idx], projection=tr_data)

        zlo = site["zoom_lon"]
        zla = site["zoom_lat"]
        ax.set_extent([zlo[0], zlo[1], zla[0], zla[1]], tr_data)

        ax.add_feature(cfeature.LAND, facecolor="black", edgecolor="black")
        ax.coastlines(resolution="50m", linewidth=0.3, color="0.3")

        gl_z = ax.gridlines(crs=tr_data, draw_labels=True, linewidth=0.15,
                            color="gray", alpha=0.3, linestyle="--")
        gl_z.xlabel_style = {"size": 6}
        gl_z.ylabel_style = {"size": 6}
        gl_z.top_labels = False
        gl_z.right_labels = False

        # Bathymetry + contours (combined for CP)
        lon_b, lat_b, z_b = load_bathy_for_site(site_key, for_overview=False)
        pcm_z = None
        if lon_b is not None:
            try:
                LON, LAT = np.meshgrid(lon_b, lat_b)
                pcm_z = ax.pcolormesh(LON, LAT, z_b, transform=tr_data,
                                     cmap="bone", vmin=-6000, vmax=0,
                                     shading="auto", alpha=0.75,
                                     rasterized=True)
                ax.contour(lon_b, lat_b, z_b,
                          levels=np.arange(-5000, 0, 1000),
                          colors="0.4", linewidths=0.3, alpha=0.5,
                          transform=tr_data)
            except Exception as e:
                print(f"    Warning zoom {site_key}: {e}")

        # Inner domain box
        draw_site_box(ax, site, tr_data, lw=2.0)

        # Decadal meander positions
        nc_key = f"nc_{site_key}"
        if nc_key in data:
            ds_nc = data[nc_key]
            lon_inner = ds_nc["longitude"].values
            # Convert meander lons to 0-360 for consistency
            lon_plot = np.where(lon_inner < 0, lon_inner + 360, lon_inner)
            months = pd.DatetimeIndex(ds_nc["month"].values)

            for lbl, ds0, ds1, dlw, dls in DECADES:
                mask = (months >= pd.Timestamp(ds0)) & (months <= pd.Timestamp(ds1))
                if mask.sum() == 0:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    ml = np.nanmean(ds_nc["center_lat"].values[mask, :], axis=0)
                v = np.isfinite(ml)
                if v.sum() > 0:
                    ax.plot(lon_plot[v], ml[v], color=site["color"],
                           lw=dlw, ls=dls, alpha=0.85, transform=tr_data,
                           solid_capstyle="round", zorder=4)

        # Panel label + title
        ax.text(0.03, 0.95, panel_labels[idx], transform=ax.transAxes,
               fontsize=11, fontweight="bold", va="top",
               bbox=dict(fc="white", ec="none", alpha=0.8))
        ax.set_title(site["name"], fontsize=8, fontweight="bold",
                    color=site["color"], pad=4)
        ax.set_xlabel("Longitude", fontsize=7)
        if idx == 0:
            ax.set_ylabel("Latitude", fontsize=7)

    # Decade legend
    decade_legs = [plt.Line2D([0], [0], color="0.3", lw=w, ls=ls, label=l)
                   for l, _, _, w, ls in DECADES]
    fig.legend(handles=decade_legs, loc="lower center", ncol=3,
              fontsize=7.5, framealpha=0.95, bbox_to_anchor=(0.5, -0.01),
              title="Decadal mean meander position", title_fontsize=8)

    fp_out = FIG_DIR / "fig1_circumpolar_map.pdf"
    fig.savefig(fp_out, dpi=300, bbox_inches="tight")
    fig.savefig(fp_out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {fp_out}")


if __name__ == "__main__":
    print("=" * 60)
    print("Figure 1: Study Region Map")
    print("=" * 60)
    data = load_all_data()
    plot_fig1(data)
    for key in list(data.keys()):
        if key.startswith("nc_") and hasattr(data[key], "close"):
            data[key].close()
    print("Done.")
