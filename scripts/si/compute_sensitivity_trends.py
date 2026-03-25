#!/usr/bin/env python3
"""
compute_sensitivity_trends.py
=============================
Compute Mann-Kendall + Sen's slope trends for all thresholds.

Reads the monthly_metrics CSVs from each threshold directory and
produces all_threshold_trends.csv for Table S1 and Figure S1.

Run on Gadi:
  source /g/data/gv90/xl1657/venvs/cmems_py311/bin/activate
  module purge
  unset PYTHONPATH
  cd /g/data/gv90/xl1657/cmems_adt/notebooks/
  python compute_sensitivity_trends.py

Author: Xinlong Liu, IMAS, University of Tasmania
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pymannkendall as mk

SENS = Path("/g/data/gv90/xl1657/cmems_adt/grl_meander_products/threshold_sensitivity")
THRESHOLDS = [15, 20, 25, 30, 35, 40]
SITES = ["CP", "PAR", "SEIR", "SWIR"]
METRICS = ["center_lat", "width_km", "speed_m_s", "eke_m2_s2"]
METRIC_LABELS = {
    "center_lat": "Position (deg lat/dec)",
    "width_km": "Width (km/dec)",
    "speed_m_s": "Speed (m/s/dec)",
    "eke_m2_s2": "EKE (m2/s2/dec)",
}

rows = []
for thresh in THRESHOLDS:
    d = SENS / f"rel{thresh:02d}"
    print(f"\n=== Threshold {thresh}% ===")
    for site in SITES:
        fp = d / f"monthly_metrics_{site}.csv"
        if not fp.exists():
            print(f"  MISSING: {fp}")
            continue

        df = pd.read_csv(fp, parse_dates=["date"])
        df = df.dropna(subset=["center_lat"])
        n_valid = len(df)

        for metric in METRICS:
            if metric not in df.columns:
                rows.append({
                    "site": site, "metric": metric, "threshold": thresh,
                    "slope_dec": float("nan"), "p_mk": float("nan"),
                    "p_mod": float("nan"), "r2": float("nan"),
                    "n_months": 0, "sign": "—", "significant": "—",
                })
                print(f"  {site} {metric}: column missing")
                continue

            s = df[metric].dropna().values
            if len(s) < 24:
                rows.append({
                    "site": site, "metric": metric, "threshold": thresh,
                    "slope_dec": float("nan"), "p_mk": float("nan"),
                    "p_mod": float("nan"), "r2": float("nan"),
                    "n_months": len(s), "sign": "—", "significant": "—",
                })
                print(f"  {site} {metric}: too few data points ({len(s)})")
                continue

            # Standard Mann-Kendall
            res = mk.original_test(s)

            # Modified Mann-Kendall (Hamed-Rao autocorrelation correction)
            try:
                mod = mk.hamed_rao_modification_test(s)
            except Exception:
                mod = res

            # Sen's slope: per month -> per decade
            slope_dec = res.slope * 12 * 10

            # R-squared from linear correlation
            rr = np.corrcoef(np.arange(len(s)), s)[0, 1]
            r2 = rr ** 2

            # Sign and significance markers
            sign = "+" if slope_dec > 0 else "−"
            sig = "**" if mod.p < 0.01 else ("*" if mod.p < 0.05 else "ns")

            rows.append({
                "site": site, "metric": metric, "threshold": thresh,
                "slope_dec": slope_dec, "p_mk": res.p, "p_mod": mod.p,
                "r2": r2, "n_months": len(s), "sign": sign,
                "significant": sig,
            })
            print(f"  {site} {metric}: slope={slope_dec:.4f}/dec, p_mod={mod.p:.4f} ({sig})")

out = pd.DataFrame(rows)
out_fp = SENS / "all_threshold_trends.csv"
out.to_csv(out_fp, index=False)
print(f"\n{'='*70}")
print(f"Saved: {out_fp}")
print(f"Total rows: {len(out)} (expected: {len(THRESHOLDS)*len(SITES)*len(METRICS)})")
print(f"{'='*70}")
