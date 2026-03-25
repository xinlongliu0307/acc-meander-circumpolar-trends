#!/usr/bin/env python3
"""Run NB02 meander detection at a specified threshold."""
import sys, gc
from pathlib import Path

# Parse command-line threshold
if len(sys.argv) < 2:
    print("Usage: python run_threshold.py <threshold_pct>")
    print("  e.g.: python run_threshold.py 15")
    sys.exit(1)

thresh_pct = int(sys.argv[1])
thresh_frac = thresh_pct / 100.0

# Set up output directory
BASE = Path("/g/data/gv90/xl1657/cmems_adt")
SENS = BASE / "grl_meander_products" / "threshold_sensitivity" / f"rel{thresh_pct:02d}"
SENS.mkdir(parents=True, exist_ok=True)

# Import NB02 core functions (won't trigger main execution due to __main__ guard)
import NB02_meander_detection as NB02

ADT_FP = BASE / "cmems_so30S_19930101_20250802_adt_sla_ugos_vgos_batch.nc"

for site_key in ["CP", "PAR", "SEIR", "SWIR"]:
    out_csv = SENS / f"monthly_metrics_{site_key}.csv"
    if out_csv.exists():
        print(f"  [{site_key}] Already exists at {thresh_pct}%, skipping.")
        continue
    print(f"\nRunning {site_key} at {thresh_pct}%...")
    df, ds = NB02.process_site(site_key, ADT_FP, SENS,
                               relat_thresh=thresh_frac, x_months=4)
    del ds
    gc.collect()

print(f"\nDone: {thresh_pct}% threshold complete.")