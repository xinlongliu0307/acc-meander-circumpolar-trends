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
def plot_fig1(data):
    print("\nGenerating Figure 1...")
    if not HAS_CARTOPY:
        print("  Skipping (no Cartopy).")
        return

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection=ccrs.SouthPolarStereo())
    ax.set_extent([-180, 180, -68, -35], ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="0.82", edgecolor="0.3", linewidth=0.5)
    ax.coastlines(resolution="50m", linewidth=0.4, color="0.3")
    tr = ccrs.PlateCarree()

    # Longitude/latitude labels
    gl = ax.gridlines(crs=tr, draw_labels=True, linewidth=0.15,
                      color="gray", alpha=0.4, linestyle="--")
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 30))
    gl.ylocator = mticker.FixedLocator(np.arange(-65, -30, 5))
    gl.xlabel_style = {"size": 7, "color": "0.4"}
    gl.ylabel_style = {"size": 7, "color": "0.4"}
    gl.top_labels = False

    # Bathymetry: ocean_deep colormap, only at 4 site regions
    for site_key in SITES:
        parts = ["CP_east", "CP_west"] if site_key == "CP" else [site_key]
        for part in parts:
            fp = GMRT_FILES.get(part)
            if fp and fp.exists():
                try:
                    lon_b, lat_b, z_b = load_gmrt(fp, coarsen=6)
                    LON, LAT = np.meshgrid(lon_b, lat_b)
                    pcm = ax.pcolormesh(LON, LAT, z_b, transform=tr,
                                       cmap="ocean", vmin=-6000, vmax=0,
                                       shading="auto", alpha=0.65,
                                       rasterized=True)
                except Exception as e:
                    print(f"    Warning: {fp.name}: {e}")

    cbar = fig.colorbar(pcm, ax=ax, orientation="horizontal",
                       fraction=0.035, pad=0.06, shrink=0.55, aspect=30)
    cbar.set_label("Depth (m)", fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    # Rounded rectangle site boxes + labels positioned away from box
    # Label positions manually tuned per site to avoid overlap
    label_positions = {
        "CP":   {"lon": 180,  "lat": -42, "ha": "center"},
        "PAR":  {"lon": -115, "lat": -43, "ha": "center"},
        "SEIR": {"lon": 141,  "lat": -40, "ha": "center"},
        "SWIR": {"lon": 30,   "lat": -39, "ha": "center"},
    }

    for site_key, site in SITES.items():
        lo0, lo1 = site["inner_lon"]
        la0, la1 = site["inner_lat"]
        col = site["color"]

        # Draw rounded-corner rectangle using many small segments
        n_seg = 60
        pad = 0.5  # padding in degrees

        if not site.get("wraps", False):
            # Simple rectangle
            lons_box = np.concatenate([
                np.linspace(lo0 - pad, lo1 + pad, n_seg),  # bottom
                np.full(n_seg, lo1 + pad),                   # right
                np.linspace(lo1 + pad, lo0 - pad, n_seg),  # top
                np.full(n_seg, lo0 - pad),                   # left
            ])
            lats_box = np.concatenate([
                np.full(n_seg, la0 - pad),                   # bottom
                np.linspace(la0 - pad, la1 + pad, n_seg),  # right
                np.full(n_seg, la1 + pad),                   # top
                np.linspace(la1 + pad, la0 - pad, n_seg),  # left
            ])
            ax.plot(lons_box, lats_box, color=col, lw=2.0, ls="-",
                   transform=tr, solid_capstyle="round")
        else:
            # Dateline crossing: two rectangles visually
            for xs, ys in [([lo0, 180], [la0, la0]), ([-180, lo1], [la0, la0]),
                           ([lo0, 180], [la1, la1]), ([-180, lo1], [la1, la1]),
                           ([lo0, lo0], [la0, la1]), ([lo1, lo1], [la0, la1])]:
                ax.plot(xs, ys, color=col, lw=2.0, ls="-", transform=tr)

        # Label
        lp = label_positions[site_key]
        ax.text(lp["lon"], lp["lat"], site["name"], fontsize=7.5,
               fontweight="bold", color=col, ha=lp["ha"], va="center",
               transform=tr,
               bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=col,
                        alpha=0.92, lw=0.8))

    # Decadal meander positions: ONLY at four sites (not circumpolar)
    for site_key in SITES:
        nc_key = f"nc_{site_key}"
        if nc_key not in data:
            continue
        ds = data[nc_key]
        lon_inner = ds["longitude"].values
        months = pd.DatetimeIndex(ds["month"].values)
        for lbl, ds0, ds1, dcol, dlw, dls in DECADES:
            mask = (months >= pd.Timestamp(ds0)) & (months <= pd.Timestamp(ds1))
            if mask.sum() == 0:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                ml = np.nanmean(ds["center_lat"].values[mask, :], axis=0)
            v = np.isfinite(ml)
            if v.sum() > 0:
                ax.plot(lon_inner[v], ml[v], color=dcol, lw=dlw,
                       ls=dls, alpha=0.85, transform=tr,
                       solid_capstyle="round")

    legend_els = [plt.Line2D([0], [0], color=c, lw=w, ls=ls, label=l)
                  for l, _, _, c, w, ls in DECADES]
    ax.legend(handles=legend_els, loc="lower left", fontsize=7.5,
             framealpha=0.95, title="Decadal mean position", title_fontsize=7.5)

    fp_out = FIG_DIR / "fig1_circumpolar_map.pdf"
    fig.savefig(fp_out, dpi=300, bbox_inches="tight")
    fig.savefig(fp_out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fp_out}")


# =====================================================================
# FIGURE 2
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
    print("GRL Figure Generation (v4 — Publication quality)")
    print("=" * 60)
    data = load_all_data()
    plot_fig1(data)
    plot_fig2(data)
    plot_fig3(data)
    plot_fig4(data)
    for key in list(data.keys()):
        if key.startswith("nc_") and hasattr(data[key], "close"):
            data[key].close()
    print(f"\nAll figures saved to: {FIG_DIR}")
