#!/usr/bin/env python3
"""
NB06_fig3.py — Figure 3: Cross-Site Comparison
Run: python NB06_fig3.py
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


def plot_fig3(data):
    print("\nGenerating Figure 3...")
    trends_df = data.get("trends")
    if trends_df is None:
        print("  Skipping.")
        return

    metrics = [
        ("center_lat", "Position\n(\u00b0/dec)", True),
        ("width_km",   "Width\n(km/dec)", False),
        ("speed_m_s",  r"Speed" + "\n" + r"(m s$^{-1}$/dec)", False),
        ("eke_m2_s2",  "EKE\n" + r"($\times 10^{-4}$ m$^{2}$ s$^{-2}$/dec)", False),
    ]
    skeys = list(SITES.keys())
    x = np.arange(len(skeys))
    bw = 0.6
    fig, axes = plt.subplots(1, 4, figsize=(13, 4))

    for j, (metric, ylabel, add_km) in enumerate(metrics):
        ax = axes[j]
        vals_for_ylim = []
        for k, sk in enumerate(skeys):
            row = trends_df[(trends_df["site"] == sk) & (trends_df["metric"] == metric)]
            if len(row) == 0:
                continue
            s = row.iloc[0]["slope_per_decade"]
            sig_flag = row.iloc[0]["significant"]
            p_val = row.iloc[0]["p_value"]
            display_s = s * 1e4 if metric == "eke_m2_s2" else s
            vals_for_ylim.append(display_s)
            col = SITES[sk]["color"]

            if sig_flag:
                ax.bar(x[k], display_s, bw, color=col, ec="black", lw=0.6)
            else:
                ax.bar(x[k], display_s, bw, fc="white", ec=col, lw=1.5, hatch="///")

        # Set ylim with padding for annotations
        if vals_for_ylim:
            ymin = min(min(vals_for_ylim), 0)
            ymax = max(max(vals_for_ylim), 0)
            rng = ymax - ymin if ymax != ymin else 1.0
            ax.set_ylim(ymin - rng * 0.35, ymax + rng * 0.35)

        # Now add annotations with correct ylim
        for k, sk in enumerate(skeys):
            row = trends_df[(trends_df["site"] == sk) & (trends_df["metric"] == metric)]
            if len(row) == 0:
                continue
            s = row.iloc[0]["slope_per_decade"]
            sig_flag = row.iloc[0]["significant"]
            p_val = row.iloc[0]["p_value"]
            display_s = s * 1e4 if metric == "eke_m2_s2" else s
            col = SITES[sk]["color"]
            star = sig_label(p_val)

            rng = ax.get_ylim()[1] - ax.get_ylim()[0]
            offset = rng * 0.04

            if add_km:
                km_val = s * DEG_TO_KM
                sign_d = "+" if s > 0 else ""
                sign_k = "+" if km_val > 0 else ""
                label_txt = f"{sign_d}{s:.2f}\u00b0\n({sign_k}{km_val:.1f} km)"
                if star:
                    label_txt += f" {star}"
                yp = display_s + offset if display_s >= 0 else display_s - offset
                va = "bottom" if display_s >= 0 else "top"
                ax.text(x[k], yp, label_txt, ha="center", va=va,
                       fontsize=5.5, color=col, fontweight="bold")
            else:
                # Significance star
                if star:
                    yp = display_s + offset if display_s >= 0 else display_s - offset
                    va = "bottom" if display_s >= 0 else "top"
                    ax.text(x[k], yp, star, ha="center", va=va,
                           fontsize=10, fontweight="bold", color=col)
                # Value annotation for all bars in site colour
                sign = "+" if display_s > 0 else ""
                if metric == "eke_m2_s2":
                    val_txt = f"{sign}{display_s:.1f}"
                elif metric == "speed_m_s":
                    val_txt = f"{sign}{display_s:.4f}"
                else:
                    val_txt = f"{sign}{display_s:.2f}"

                # Place value inside or adjacent to bar
                if star:
                    yp2 = yp + offset * 1.5 if display_s >= 0 else yp - offset * 1.5
                else:
                    yp2 = display_s + offset if display_s >= 0 else display_s - offset
                va2 = "bottom" if display_s >= 0 else "top"
                ax.text(x[k], yp2, val_txt, ha="center", va=va2,
                       fontsize=5.5, color=col, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([SITES[sk]["name"] for sk in skeys],
                          fontsize=6, rotation=35, ha="right")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(f"({chr(97+j)})", fontsize=10, fontweight="bold", loc="left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout(w_pad=1.0)
    fp_out = FIG_DIR / "fig3_crosssite_comparison.pdf"
    fig.savefig(fp_out, dpi=300, bbox_inches="tight")
    fig.savefig(fp_out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fp_out}")


# =====================================================================
# FIGURE 4
# =====================================================================


if __name__ == "__main__":
    print("=" * 60)
    print("Figure 3: Cross-Site Comparison")
    print("=" * 60)
    data = load_all_data()
    plot_fig3(data)
    for key in list(data.keys()):
        if key.startswith("nc_") and hasattr(data[key], "close"):
            data[key].close()
    print("Done.")
