#!/usr/bin/env python3
"""
NB06_fig4.py — Figure 4: Forcing Context
Run: python NB06_fig4.py
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


def plot_fig4(data):
    print("\nGenerating Figure 4...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    all_years = list(range(1993, 2026))

    # ── Panel (a) ──
    ax = axes[0]
    wt = data.get("wind_trends")
    for sk, site in SITES.items():
        wk = f"wind_{sk}"
        if wk not in data or "u10" not in data[wk].columns:
            continue
        u_ann = data[wk]["u10"].resample("YS").mean()
        ax.plot(u_ann.index, u_ann.values, color=site["color"], lw=1.2,
               label=site["name"], marker="o", ms=2.5, zorder=3)
        if wt is not None:
            row = wt[(wt["site"] == sk) & (wt["metric"] == "u10")]
            if len(row) > 0:
                slope = row.iloc[0]["slope_per_decade"]
                sig_flag = row.iloc[0]["significant"]
                xn = np.arange(len(u_ann))
                ax.plot(u_ann.index,
                       u_ann.values[0] + (slope / 10) * xn,
                       color=site["color"], lw=0.8,
                       ls="-" if sig_flag else "--", alpha=0.45, zorder=2)

    # Annotate wind trends — place below the data in the lower portion
    if wt is not None:
        u10r = wt[wt["metric"] == "u10"].sort_values("site")
        for idx, (_, row) in enumerate(u10r.iterrows()):
            sk = row["site"]
            slope = row["slope_per_decade"]
            p = row["p_value"]
            star = sig_label(p)
            ann = f"{SITES[sk]['name']}: +{slope:.3f}" + r" m s$^{-1}$/dec"
            if star:
                ann += f" {star}"
            # Place at bottom-left, stacked vertically
            ax.text(0.02, 0.22 - idx * 0.06, ann,
                   transform=ax.transAxes, fontsize=6.5,
                   color=SITES[sk]["color"], fontweight="bold",
                   bbox=dict(fc="white", ec="none", alpha=0.85))

    ax.set_ylabel(r"Zonal wind speed (m s$^{-1}$)", fontsize=9)
    ax.set_xlabel("Year", fontsize=9)
    ax.set_title("(a)", fontsize=10, fontweight="bold", loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xticks([pd.Timestamp(f"{y}-01-01") for y in all_years])
    ax.set_xticklabels([str(y) for y in all_years], rotation=45,
                       ha="right", fontsize=5.5)

    # Legend below x-axis
    ax.legend(fontsize=6.5, loc="upper center",
             bbox_to_anchor=(0.5, -0.22), ncol=2,
             framealpha=0.95, columnspacing=1.0)

    # ── Panel (b) ──
    ax = axes[1]
    argo = data.get("argo")
    if argo is not None and len(argo) > 0:
        sks = argo["site"].values
        warming = argo["mean_warming_C"].values
        grad = argo["mean_abs_grad_change_C_100km"].values
        cols = [SITES[k]["color"] for k in sks]
        xp = np.arange(len(sks))
        bw = 0.35

        ax.bar(xp - bw/2, warming, bw, color=cols, ec="black", lw=0.5, alpha=0.85)
        ax2 = ax.twinx()
        ax2.bar(xp + bw/2, grad, bw, color=cols, ec="black", lw=0.5,
               alpha=0.4, hatch="...")

        for k in range(len(sks)):
            ax.text(xp[k] - bw/2, warming[k] + 0.003,
                   f"{warming[k]:.3f}", ha="center", va="bottom", fontsize=6.5,
                   color=cols[k], fontweight="bold")
            ax2.text(xp[k] + bw/2, grad[k] + 0.001,
                    f"{grad[k]:.3f}", ha="center", va="bottom", fontsize=6.5,
                    color=cols[k], fontweight="bold")

        ax.set_xticks(xp)
        ax.set_xticklabels([SITES[k]["name"] for k in sks],
                          fontsize=7, rotation=25, ha="right")
        ax.set_ylabel("Warming at 500 dbar (\u00b0C)", fontsize=9)
        ax2.set_ylabel(
            r"$|\partial \overline{T}/\partial y|$ change ($^\circ$C / 100 km)",
            fontsize=9)
        ax.set_title("(b)", fontsize=10, fontweight="bold", loc="left")

        l1 = mpatches.Patch(fc="gray", ec="black", alpha=0.85,
                           label="Warming (\u00b0C)")
        l2 = mpatches.Patch(fc="gray", ec="black", alpha=0.4, hatch="...",
                           label=r"$|\partial \overline{T}/\partial y|$ change")
        ax.legend(handles=[l1, l2], fontsize=7, loc="upper left")
        ax.spines["top"].set_visible(False)
        ax2.spines["top"].set_visible(False)

    fig.tight_layout(w_pad=3.0)
    fig.subplots_adjust(bottom=0.20)

    fp_out = FIG_DIR / "fig4_forcing_context.pdf"
    fig.savefig(fp_out, dpi=300, bbox_inches="tight")
    fig.savefig(fp_out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fp_out}")


# =====================================================================


if __name__ == "__main__":
    print("=" * 60)
    print("Figure 4: Forcing Context")
    print("=" * 60)
    data = load_all_data()
    plot_fig4(data)
    for key in list(data.keys()):
        if key.startswith("nc_") and hasattr(data[key], "close"):
            data[key].close()
    print("Done.")
