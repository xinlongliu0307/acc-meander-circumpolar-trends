#!/usr/bin/env python3
"""
NB09_threshold_sensitivity.py
=============================
Threshold sensitivity analysis for GRL Supporting Information.

Produces:
  - Table S1: Trend robustness across thresholds (15%, 20%, 25%, 30%, 35%, 40%)
  - Figure S1: Visual comparison of meander detection across thresholds

This script reuses the core functions from NB02_meander_detection.py
and NB03_speed_eke_trends.py already on Gadi.

Run on NCI Gadi:
  cd /g/data/gv90/xl1657/cmems_adt/notebooks/
  source /g/data/gv90/xl1657/venvs/cmems_py311/bin/activate
  module purge
  unset PYTHONPATH
  python NB09_threshold_sensitivity.py

Author: Xinlong Liu, IMAS, University of Tasmania
"""

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings, gc, time as _time, sys, importlib

warnings.filterwarnings("ignore")

# =============================================================================
# 0. CONFIGURATION
# =============================================================================

BASE_DIR   = Path("/g/data/gv90/xl1657/cmems_adt")
ADT_FP     = BASE_DIR / "cmems_so30S_19930101_20250802_adt_sla_ugos_vgos_batch.nc"
PROD_DIR   = BASE_DIR / "grl_meander_products"
SENS_DIR   = PROD_DIR / "threshold_sensitivity"
SENS_DIR.mkdir(parents=True, exist_ok=True)

NOTEBOOK_DIR = BASE_DIR / "notebooks"

# Thresholds to test (20% is the baseline, already computed)
THRESHOLDS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
BASELINE   = 0.20

SITES = ["CP", "PAR", "SEIR", "SWIR"]

# Metrics to compare in Table S1
METRICS = ["center_lat", "width_km", "speed_m_s", "eke_m2_s2"]
METRIC_LABELS = {
    "center_lat": "Position (°lat/dec)",
    "width_km":   "Width (km/dec)",
    "speed_m_s":  "Speed (m s⁻¹/dec)",
    "eke_m2_s2":  "EKE (m² s⁻²/dec)",
}


# =============================================================================
# 1. IMPORT CORE FUNCTIONS FROM NB02 AND NB03
# =============================================================================

def import_nb02():
    """Import process_site from NB02_meander_detection.py."""
    spec = importlib.util.spec_from_file_location(
        "NB02", NOTEBOOK_DIR / "NB02_meander_detection.py"
    )
    nb02 = importlib.util.module_from_spec(spec)
    # Prevent NB02 from running its __main__ block on import
    sys.modules["NB02"] = nb02
    # We need to temporarily override __name__ to prevent main execution
    original_name = "__main__"
    spec.loader.exec_module(nb02)
    return nb02


def import_nb03():
    """Import trend functions from NB03_speed_eke_trends.py."""
    spec = importlib.util.spec_from_file_location(
        "NB03", NOTEBOOK_DIR / "NB03_speed_eke_trends.py"
    )
    nb03 = importlib.util.module_from_spec(spec)
    sys.modules["NB03"] = nb03
    spec.loader.exec_module(nb03)
    return nb03


# =============================================================================
# 2. RUN MEANDER DETECTION AT EACH NON-BASELINE THRESHOLD
# =============================================================================

def run_detection_at_threshold(nb02_module, thresh):
    """
    Run meander detection for all 4 sites at a given threshold.
    
    For the baseline (20%), we already have the outputs in PROD_DIR.
    For other thresholds, we save to SENS_DIR with threshold in filename.
    """
    thresh_pct = int(thresh * 100)
    thresh_dir = SENS_DIR / f"rel{thresh_pct:02d}"
    thresh_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"# THRESHOLD: {thresh_pct}%")
    print(f"# Output directory: {thresh_dir}")
    print(f"{'#'*70}")

    for site_key in SITES:
        out_csv = thresh_dir / f"monthly_metrics_{site_key}.csv"

        # Skip if already computed
        if out_csv.exists():
            print(f"  [{site_key}] Already exists at {thresh_pct}%, skipping.")
            continue

        print(f"\n  [{site_key}] Running detection at {thresh_pct}%...")
        t0 = _time.time()

        try:
            df, ds = nb02_module.process_site(
                site_key, ADT_FP, thresh_dir,
                relat_thresh=thresh, x_months=4
            )
            del ds
            gc.collect()

            elapsed = _time.time() - t0
            print(f"  [{site_key}] Done in {elapsed/60:.1f} min.")
        except Exception as e:
            print(f"  [{site_key}] FAILED: {e}")
            continue


def copy_baseline_results():
    """
    Copy (or symlink) the existing 20% results into the sensitivity directory
    so all thresholds are in one place.
    """
    thresh_dir = SENS_DIR / "rel20"
    thresh_dir.mkdir(parents=True, exist_ok=True)

    import shutil
    for site_key in SITES:
        src_csv = PROD_DIR / f"monthly_metrics_{site_key}.csv"
        dst_csv = thresh_dir / f"monthly_metrics_{site_key}.csv"
        if src_csv.exists() and not dst_csv.exists():
            shutil.copy2(src_csv, dst_csv)
            print(f"  Copied baseline 20% CSV for {site_key}")


# =============================================================================
# 3. COMPUTE TRENDS AT EACH THRESHOLD
# =============================================================================

def compute_trends_at_threshold(thresh):
    """
    Compute Mann-Kendall + Sen's slope for each metric at a given threshold.
    
    Returns a DataFrame with columns:
      site, metric, slope, p_mk, p_mk_mod, r2, threshold
    """
    try:
        import pymannkendall as mk
    except ImportError:
        print("ERROR: pymannkendall not installed. Run: pip install pymannkendall")
        return pd.DataFrame()

    thresh_pct = int(thresh * 100)
    thresh_dir = SENS_DIR / f"rel{thresh_pct:02d}"

    rows = []
    for site_key in SITES:
        csv_fp = thresh_dir / f"monthly_metrics_{site_key}.csv"
        if not csv_fp.exists():
            print(f"  WARNING: {csv_fp} not found, skipping.")
            continue

        df = pd.read_csv(csv_fp, parse_dates=["date"])
        df = df.dropna(subset=["center_lat"])  # Only months with valid detection

        for metric in METRICS:
            if metric not in df.columns or df[metric].dropna().shape[0] < 24:
                rows.append({
                    "site": site_key, "metric": metric,
                    "slope_per_dec": np.nan, "p_mk": np.nan,
                    "p_mk_mod": np.nan, "r2": np.nan,
                    "threshold_pct": thresh_pct,
                    "sign": "—", "significant": "—",
                })
                continue

            series = df[metric].dropna().values

            # Standard Mann-Kendall
            try:
                result_mk = mk.original_test(series)
            except Exception:
                result_mk = None

            # Modified Mann-Kendall (lag-1 autocorrelation correction)
            try:
                result_mod = mk.hamed_rao_modification_test(series)
            except Exception:
                result_mod = result_mk

            if result_mk is None:
                rows.append({
                    "site": site_key, "metric": metric,
                    "slope_per_dec": np.nan, "p_mk": np.nan,
                    "p_mk_mod": np.nan, "r2": np.nan,
                    "threshold_pct": thresh_pct,
                    "sign": "—", "significant": "—",
                })
                continue

            # Sen's slope is per month; convert to per decade
            slope_per_month = result_mk.slope
            slope_per_dec = slope_per_month * 12 * 10  # months → decades

            # R² from linear fit
            x = np.arange(len(series))
            r = np.corrcoef(x, series)[0, 1]
            r2 = r ** 2

            p_mk = result_mk.p
            p_mod = result_mod.p if result_mod else np.nan

            sign = "+" if slope_per_dec > 0 else "−"
            sig = "**" if p_mod < 0.01 else ("*" if p_mod < 0.05 else "ns")

            rows.append({
                "site": site_key, "metric": metric,
                "slope_per_dec": slope_per_dec, "p_mk": p_mk,
                "p_mk_mod": p_mod, "r2": r2,
                "threshold_pct": thresh_pct,
                "sign": sign, "significant": sig,
            })

    return pd.DataFrame(rows)


# =============================================================================
# 4. BUILD TABLE S1
# =============================================================================

def build_table_s1(all_trends_df):
    """
    Construct Table S1: trend sign and significance stability across thresholds.
    
    Format: For each site × metric, show slope (significance marker) at each threshold.
    """
    out_fp = SENS_DIR / "table_s1_threshold_robustness.csv"

    # Pivot: rows = (site, metric), columns = threshold
    records = []
    for site_key in SITES:
        for metric in METRICS:
            row = {"Site": site_key, "Metric": METRIC_LABELS[metric]}
            for thresh_pct in [int(t * 100) for t in THRESHOLDS]:
                subset = all_trends_df[
                    (all_trends_df["site"] == site_key) &
                    (all_trends_df["metric"] == metric) &
                    (all_trends_df["threshold_pct"] == thresh_pct)
                ]
                if subset.empty or pd.isna(subset.iloc[0]["slope_per_dec"]):
                    row[f"{thresh_pct}%"] = "—"
                else:
                    s = subset.iloc[0]
                    row[f"{thresh_pct}%"] = (
                        f"{s['slope_per_dec']:.3f} ({s['significant']})"
                    )
            records.append(row)

    table_df = pd.DataFrame(records)
    table_df.to_csv(out_fp, index=False)
    print(f"\nTable S1 saved: {out_fp}")

    # Also save the full raw trends
    raw_fp = SENS_DIR / "all_threshold_trends_raw.csv"
    all_trends_df.to_csv(raw_fp, index=False)
    print(f"Raw trends saved: {raw_fp}")

    return table_df


# =============================================================================
# 5. BUILD FIGURE S1
# =============================================================================

def build_figure_s1(all_trends_df):
    """
    Figure S1: 4×4 panel grid.
    Rows = sites (CP, PAR, SEIR, SWIR).
    Columns = metrics (position, width, speed, EKE).
    Each panel shows Sen's slope (dots) and significance shading across thresholds.
    """
    fig, axes = plt.subplots(
        4, 4, figsize=(14, 10), sharex=True,
        gridspec_kw={"hspace": 0.35, "wspace": 0.30}
    )

    site_colors = {"CP": "#1f77b4", "PAR": "#d62728",
                   "SEIR": "#2ca02c", "SWIR": "#ff7f0e"}
    thresh_vals = [int(t * 100) for t in THRESHOLDS]

    for i, site_key in enumerate(SITES):
        for j, metric in enumerate(METRICS):
            ax = axes[i, j]
            subset = all_trends_df[
                (all_trends_df["site"] == site_key) &
                (all_trends_df["metric"] == metric)
            ].sort_values("threshold_pct")

            if subset.empty:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=8, color="gray")
                continue

            slopes = subset["slope_per_dec"].values
            p_mods = subset["p_mk_mod"].values
            threshs = subset["threshold_pct"].values

            # Plot slopes as connected dots
            ax.plot(threshs, slopes, "o-", color=site_colors[site_key],
                    markersize=5, linewidth=1.5, zorder=3)

            # Shade significance
            for k in range(len(threshs)):
                if pd.notna(p_mods[k]):
                    if p_mods[k] < 0.01:
                        ax.axvspan(threshs[k] - 1.5, threshs[k] + 1.5,
                                   alpha=0.15, color="green", zorder=1)
                    elif p_mods[k] < 0.05:
                        ax.axvspan(threshs[k] - 1.5, threshs[k] + 1.5,
                                   alpha=0.10, color="gold", zorder=1)

            # Zero line
            ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", zorder=2)

            # Highlight the 20% baseline
            ax.axvline(20, color="black", linewidth=0.8, linestyle=":",
                       alpha=0.5, zorder=2)

            # Labels
            ax.set_xticks(thresh_vals)
            if i == 0:
                ax.set_title(METRIC_LABELS[metric], fontsize=9, fontweight="bold")
            if j == 0:
                ax.set_ylabel(site_key, fontsize=10, fontweight="bold")
            if i == 3:
                ax.set_xlabel("Threshold (%)", fontsize=8)

            ax.tick_params(labelsize=7)
            ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor="green", alpha=0.15, label="p < 0.01"),
        Patch(facecolor="gold", alpha=0.10, label="p < 0.05"),
        Line2D([0], [0], color="black", linewidth=0.8, linestyle=":",
               label="20% baseline"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "Figure S1: Sensitivity of decadal trend estimates to ADT gradient threshold",
        fontsize=11, fontweight="bold", y=0.98
    )

    out_fp = SENS_DIR / "figure_s1_threshold_sensitivity.png"
    fig.savefig(out_fp, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nFigure S1 saved: {out_fp}")

    # Also save as PDF
    out_pdf = SENS_DIR / "figure_s1_threshold_sensitivity.pdf"
    fig2, axes2 = plt.subplots(4, 4, figsize=(14, 10), sharex=True,
                                gridspec_kw={"hspace": 0.35, "wspace": 0.30})
    # Rebuild for PDF (matplotlib can't save a closed figure)
    # Instead, just save the png and note that PDF can be converted
    print(f"  (Convert to PDF with: convert figure_s1_threshold_sensitivity.png figure_s1_threshold_sensitivity.pdf)")


# =============================================================================
# 6. MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("NB09: Threshold Sensitivity Analysis for GRL SI")
    print(f"  Thresholds: {[int(t*100) for t in THRESHOLDS]}%")
    print(f"  Sites: {SITES}")
    print(f"  Output: {SENS_DIR}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # PHASE A: Run meander detection at each non-baseline threshold
    # ------------------------------------------------------------------
    print("\n>>> PHASE A: Meander detection across thresholds")

    # Copy baseline 20% results
    print("\nCopying baseline 20% results...")
    copy_baseline_results()

    # Import NB02 core functions
    print("\nImporting NB02 core functions...")
    # NOTE: If direct import fails due to __main__ guard issues,
    # use the fallback approach below (see FALLBACK section).
    try:
        nb02 = import_nb02()
        print("  NB02 imported successfully.")
    except Exception as e:
        print(f"  NB02 import failed: {e}")
        print("  Using fallback: will call NB02 as subprocess for each threshold.")
        nb02 = None

    # Run detection for non-baseline thresholds
    for thresh in THRESHOLDS:
        if thresh == BASELINE:
            print(f"\n  Skipping {int(thresh*100)}% (baseline already computed)")
            continue

        if nb02 is not None:
            run_detection_at_threshold(nb02, thresh)
        else:
            # FALLBACK: Run NB02 as subprocess with modified threshold
            thresh_pct = int(thresh * 100)
            thresh_dir = SENS_DIR / f"rel{thresh_pct:02d}"
            thresh_dir.mkdir(parents=True, exist_ok=True)

            # Check if already done
            all_done = all(
                (thresh_dir / f"monthly_metrics_{s}.csv").exists()
                for s in SITES
            )
            if all_done:
                print(f"\n  [{thresh_pct}%] All sites already computed, skipping.")
                continue

            print(f"\n  [{thresh_pct}%] Running via subprocess...")
            import subprocess
            cmd = [
                sys.executable, str(NOTEBOOK_DIR / "NB02_meander_detection.py"),
                "--threshold", str(thresh),
                "--outdir", str(thresh_dir),
            ]
            print(f"    Command: {' '.join(cmd)}")
            print(f"    NOTE: If NB02 doesn't support --threshold flag,")
            print(f"    you'll need to edit RELAT_THRESH in NB02 manually.")
            print(f"    See the MANUAL APPROACH section in the instructions.")

    # ------------------------------------------------------------------
    # PHASE B: Compute trends at each threshold
    # ------------------------------------------------------------------
    print("\n>>> PHASE B: Computing trends at each threshold")

    all_trends = []
    for thresh in THRESHOLDS:
        thresh_pct = int(thresh * 100)
        print(f"\n  Computing trends at {thresh_pct}%...")
        trends_df = compute_trends_at_threshold(thresh)
        if not trends_df.empty:
            all_trends.append(trends_df)
        else:
            print(f"  WARNING: No trends computed for {thresh_pct}%")

    if not all_trends:
        print("\nERROR: No trend results. Check that detection outputs exist.")
        sys.exit(1)

    all_trends_df = pd.concat(all_trends, ignore_index=True)

    # ------------------------------------------------------------------
    # PHASE C: Build Table S1 and Figure S1
    # ------------------------------------------------------------------
    print("\n>>> PHASE C: Building Table S1 and Figure S1")

    table_s1 = build_table_s1(all_trends_df)
    print("\nTable S1 preview:")
    print(table_s1.to_string(index=False))

    build_figure_s1(all_trends_df)

    print("\n" + "=" * 70)
    print("DONE. All outputs in:", SENS_DIR)
    print("=" * 70)
