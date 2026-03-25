#!/usr/bin/env python3
"""
NB06_fig2.py — Figure 2: Trend Time Series
Run: python NB06_fig2.py
"""

#!/usr/bin/env python3
"""
NB06_figures.py (v4 — GRL publication-quality)
===============================================
"""

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
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
    "CP":   {"name": "Campbell Plateau",        "short": "CP",   "color": CB["CP"],
             "inner_lon": (150, -150), "inner_lat": (-57, -46), "wraps": True,
             "outer_lon": (150, -150), "outer_lat": (-70, -30)},
    "PAR":  {"name": "Pacific-Antarctic Ridge",  "short": "PAR",  "color": CB["PAR"],
             "inner_lon": (-150, -80), "inner_lat": (-60, -48), "wraps": False,
             "outer_lon": (-150, -60), "outer_lat": (-70, -30)},
    "SEIR": {"name": "Southeast Indian Ridge",   "short": "SEIR", "color": CB["SEIR"],
             "inner_lon": (130, 152),  "inner_lat": (-56, -44), "wraps": False,
             "outer_lon": (120, 155),  "outer_lat": (-65, -30)},
    "SWIR": {"name": "Southwest Indian Ridge",   "short": "SWIR", "color": CB["SWIR"],
             "inner_lon": (15, 45),    "inner_lat": (-58, -44), "wraps": False,
             "outer_lon": (5, 55),     "outer_lat": (-65, -35)},
}

GMRT_FILES = {
    "CP_east": GMRT_DIR / "GMRTv4_4_1_20260314topo_CP_East.grd",
    "CP_west": GMRT_DIR / "GMRTv4_4_1_20260314topo_CP_West.grd",
    "PAR":     GMRT_DIR / "GMRTv4_4_1_20260314topo_PAR.grd",
    "SEIR":    GMRT_DIR / "GMRTv4_4_1_20260314topo_SEIR.grd",
    "SWIR":    GMRT_DIR / "GMRTv4_4_1_20260314topo_SWIR.grd",
}

DECADES = [
    ("1993\u20132002", "1993-01", "2002-12", "#4575b4", 1.0, "-"),
    ("2003\u20132014", "2003-01", "2014-12", "#7570b3", 1.6, "--"),
    ("2015\u20132025", "2015-01", "2025-12", "#d95f02", 2.2, "-"),
]

DEG_TO_KM = 111.32

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.linewidth": 0.8, "axes.labelsize": 10,
    "axes.titlesize": 11, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "legend.framealpha": 0.9, "figure.dpi": 150,
    "mathtext.default": "regular",
})


def sig_label(p_val):
    return "*" if p_val < 0.05 else ""


def load_all_data():
    print("Loading data...")
    data = {}
    for key in SITES:
        fp = BASE_DIR / f"monthly_metrics_{key}.csv"
        if fp.exists():
            data[key] = pd.read_csv(fp, index_col=0, parse_dates=True)
            print(f"  {key}: {len(data[key])} months")
    for name, tag in [("trend_results_mannkendall.csv", "trends"),
                      ("era5_wind_trends.csv", "wind_trends"),
                      ("argo_temperature_summary.csv", "argo")]:
        fp = BASE_DIR / name
        if fp.exists():
            data[tag] = pd.read_csv(fp)
    for key in SITES:
        fp = BASE_DIR / f"era5_wind_{key}.csv"
        if fp.exists():
            data[f"wind_{key}"] = pd.read_csv(fp, index_col=0, parse_dates=True)
    for key in SITES:
        fp = BASE_DIR / f"meander_detection_{key}_rel20_x4m.nc"
        if fp.exists():
            data[f"nc_{key}"] = xr.open_dataset(fp)
    return data


def load_gmrt(fp, coarsen=4):
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


# =====================================================================
# FIGURE 1
# =====================================================================


def plot_fig2(data):
    print("\nGenerating Figure 2...")
    metrics = [
        ("center_lat", "Degree of latitude", "\u00b0/dec", True),
        ("width_km",   "Width anomaly (km)", "km/dec", False),
        ("speed_m_s",  r"Speed anomaly (m s$^{-1}$)", r"m s$^{-1}$/dec", False),
        ("eke_m2_s2",  r"EKE anomaly (m$^{2}$ s$^{-2}$)", r"m$^{2}$ s$^{-2}$/dec", False),
    ]
    trends_df = data.get("trends")
    fig, axes = plt.subplots(4, 1, figsize=(8, 12), sharex=True)
    all_years = list(range(1993, 2026))

    for j, (metric, ylabel, unit_str, add_km) in enumerate(metrics):
        ax = axes[j]
        anns = []
        for site_key, site in SITES.items():
            if site_key not in data or metric not in data[site_key].columns:
                continue
            y = data[site_key][metric].dropna()
            if len(y) == 0:
                continue
            y_anom = y - y.groupby(y.index.month).transform("mean")
            y_ann = y_anom.resample("YS").mean()
            ax.plot(y_ann.index, y_ann.values, color=site["color"], lw=1.3,
                   label=site["name"], marker="o", ms=2.5, zorder=3)

            if trends_df is not None:
                row = trends_df[(trends_df["site"] == site_key) &
                               (trends_df["metric"] == metric)]
                if len(row) > 0:
                    slope = row.iloc[0]["slope_per_decade"]
                    p_val = row.iloc[0]["p_value"]
                    mid = len(y_ann) // 2
                    x_yr = np.arange(len(y_ann))
                    ax.plot(y_ann.index, (slope / 10) * (x_yr - mid),
                           color=site["color"], lw=1.0, ls="--", alpha=0.5,
                           zorder=2)
                    sig = sig_label(p_val)
                    sign = "+" if slope > 0 else ""
                    if add_km:
                        km_val = slope * DEG_TO_KM
                        km_sign = "+" if km_val > 0 else ""
                        txt = f"{site['name']}: {sign}{slope:.2f} {unit_str} ({km_sign}{km_val:.1f} km/dec)"
                    else:
                        txt = f"{site['name']}: {sign}{slope:.3g} {unit_str}"
                    if sig:
                        txt += f" {sig}"
                    anns.append((txt, site["color"]))

        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.text(0.015, 0.93, f"({chr(97+j)})", transform=ax.transAxes,
               fontsize=12, fontweight="bold", va="top")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Statistics box below x-axis using figure coordinates
        if anns:
            ann_lines = []
            for txt, col in anns:
                ann_lines.append((txt, col))
            # Store for later placement
            ax._stat_annotations = ann_lines

    # X-axis
    axes[-1].set_xlabel("Year", fontsize=10)
    axes[-1].set_xticks([pd.Timestamp(f"{y}-01-01") for y in all_years])
    axes[-1].set_xticklabels([str(y) for y in all_years], rotation=45,
                             ha="right", fontsize=6)

    fig.tight_layout(h_pad=1.8, rect=[0, 0.10, 1, 1])

    # Place stat annotations below each subplot using axes coordinates
    for j, ax in enumerate(axes):
        if hasattr(ax, "_stat_annotations"):
            anns = ax._stat_annotations
            for idx, (txt, col) in enumerate(anns):
                # Place below the axis, using negative y in axes coords
                y_pos = -0.18 - idx * 0.09
                ax.text(1.0, y_pos, txt, transform=ax.transAxes, fontsize=6,
                       va="top", ha="right", color=col, fontweight="bold",
                       family="monospace",
                       bbox=dict(boxstyle="round,pad=0.1", fc="white",
                                ec="none", alpha=0.9) if idx == 0 else {})

    # Legend at very bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=7.5,
              framealpha=0.95, bbox_to_anchor=(0.5, 0.0),
              columnspacing=1.5, handlelength=2.0)

    fp_out = FIG_DIR / "fig2_trend_timeseries.pdf"
    fig.savefig(fp_out, dpi=300, bbox_inches="tight")
    fig.savefig(fp_out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fp_out}")


# =====================================================================
# FIGURE 3
# =====================================================================


if __name__ == "__main__":
    print("=" * 60)
    print("Figure 2: Trend Time Series")
    print("=" * 60)
    data = load_all_data()
    plot_fig2(data)
    for key in list(data.keys()):
        if key.startswith("nc_") and hasattr(data[key], "close"):
            data[key].close()
    print("Done.")
