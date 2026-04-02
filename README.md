# Circumpolar Satellite Evidence for Topographically-Modulated Multi-Decadal Evolution of Southern Ocean Standing Meanders

[![DOI](https://img.shields.io/badge/DOI-10.1029%2F2023JC019876-blue)](https://doi.org/10.1029/2023JC019876)
[![DOI](https://img.shields.io/badge/DOI-10.1029%2F2025JC023527-blue)](https://doi.org/10.1029/2025JC023527)
[![Data](https://img.shields.io/badge/Data-CMEMS%20DUACS%20L4-green)](https://data.marine.copernicus.eu/)
[![HPC](https://img.shields.io/badge/HPC-NCI%20Gadi-orange)](https://nci.org.au/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Analysis code for the manuscript submitted to **Geophysical Research Letters (GRL)**:

> **Liu, X.** (2026). Circumpolar Satellite Evidence for Topographically-Modulated Multi-Decadal Evolution of Southern Ocean Standing Meanders. *Submitted to Geophysical Research Letters*.

---

## Key Findings

The first **circumpolar, multi-decadal (1993–2025) observational synthesis** of standing meander structural evolution in the Antarctic Circumpolar Current (ACC), using satellite altimetry across four topographically controlled sites:

| Site | Abbr. | Topographic Control |
|------|:-----:|---------------------|
| Campbell Plateau | **CP** | Plateau deflection at ~170°E |
| Pacific-Antarctic Ridge | **PAR** | Ridge deflection at ~150°W–80°W |
| Southeast Indian Ridge | **SEIR** | Ridge interaction at ~130°E–152°E |
| Southwest Indian Ridge | **SWIR** | Ridge interaction at ~15°E–45°E |

**Principal results (corrected EKE pipeline — 12 of 16 metric–site trends significant at *p* < 0.05):**

- **Speed intensification is universal**: All four meanders accelerate (+2.8 to +7.9 × 10⁻³ m s⁻¹ dec⁻¹, *p* < 0.001).
- **EKE trends are uniformly positive**: All sites show significant eddy kinetic energy increases.
- **Width changes are site-dependent**: PAR widens (+4.1 km/dec), while CP and SEIR narrow.
- **SWIR poleward shift**: Only SWIR shows a significant positional trend (−0.11°/dec, *p* = 0.03).

## Repository Structure

```
acc-meander-circumpolar-trends/
├── scripts/
│   ├── core/                          # Main analysis pipeline
│   │   ├── NB02_meander_detection.py       # ADT gradient-based meander detection
│   │   ├── NB02_patch_width.py             # Half-peak-height width recomputation
│   │   ├── NB03_speed_eke_trends.py        # Speed, EKE + Mann-Kendall trends
│   │   ├── NB04_argo_temperature.py        # Argo 500 dbar temperature analysis
│   │   ├── NB05a_era5_download.py          # ERA5 wind data download (CDS API)
│   │   └── NB05_era5_wind.py               # ERA5 wind trend analysis
│   ├── si/                            # Supporting Information generation
│   │   ├── NB09_threshold_sensitivity.py   # Threshold robustness (Fig S1, Table S1)
│   │   ├── run_threshold.py                # CLI runner for single-threshold detection
│   │   ├── patch_width_all_thresholds.py   # Half-peak-height patch for all thresholds
│   │   ├── compute_sensitivity_trends.py   # Mann-Kendall trends across thresholds
│   │   ├── NB10_resolution_metric_comparison.py  # Resolution/metric comparison (Fig S2, Table S2)
│   │   └── NB11_domain_map.py              # Domain definition map (Fig S3)
│   └── utils/                         # Utility scripts
│       └── combine_cp_bathymetry.py        # Combine CP GMRT tiles across dateline
├── figures/                           # Publication figure generation
│   ├── NB06_figures.py                     # All 4 figures (combined script)
│   ├── NB06_fig1.py                        # Figure 1: Circumpolar map + zoom panels
│   ├── NB06_fig2.py                        # Figure 2: Trend time series
│   ├── NB06_fig3.py                        # Figure 3: Cross-site bar comparison
│   └── NB06_fig4.py                        # Figure 4: Forcing context (wind + Argo)
├── notebooks/                         # Jupyter notebooks (interactive exploration)
│   ├── NB02_meander_detection.ipynb
│   ├── NB03_speed_eke_trends.ipynb
│   └── NB06_figures.ipynb
├── docs/
│   └── PIPELINE.md                         # Detailed execution guide
├── environment.yml                         # Conda environment specification
├── LICENSE                                 # MIT License
└── README.md
```

## Pipeline Overview

```
CMEMS ADT (0.125° daily, 1993–2025)
        │
        ▼
┌──────────────────────────────┐
│  NB02: Meander Detection      │  ADT gradient → 20% threshold → 4-month occurrence
│  NB02_patch_width             │  Half-peak-height width correction
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  NB03: Speed + EKE + Trends   │  Per-grid-point EKE: spatial_mean(0.5(u'²+v'²))
│                                │  Modified MK (Hamed & Rao 1998) when |ACF₁| > 0.1
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  NB04: Argo Temperature       │  500 dbar warming + ∂T/∂y gradient changes
│  NB05: ERA5 Winds             │  Zonal wind + wind speed trends (1993–2025)
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  NB06: Figures 1–4            │  GRL publication-quality figures
└──────────────────────────────┘
           │
           ▼
┌──────────────────────────────┐
│  Supporting Information        │
│  NB09: Threshold sensitivity   │  Fig S1, Table S1 (15–40% thresholds)
│  NB10: Resolution comparison   │  Fig S2, Table S2 (CP + PAR, 4 conditions)
│  NB11: Domain definition map   │  Fig S3
└──────────────────────────────┘
```

## Data Sources

| Dataset | Source | Resolution | Period |
|---------|--------|-----------|--------|
| CMEMS DUACS L4 ADT | Copernicus Marine Service | 0.125° daily | Jan 1993 – Aug 2025 |
| ERA5 10m winds | C3S Climate Data Store | 0.25° monthly | 1993–2025 |
| Roemmich-Gilson Argo | Scripps Institution | 1° monthly | 2004–2025 |
| GMRT Bathymetry | Global Multi-Resolution Topography | variable | — |

## Getting Started

### Prerequisites

Designed for **NCI Gadi** but adaptable to any HPC with the datasets above.

```bash
# On NCI Gadi
source /g/data/gv90/xl1657/venvs/cmems_py311/bin/activate

# Or create a fresh environment
conda env create -f environment.yml
conda activate acc-meander
```

### Execution Order

```bash
# === Core pipeline ===
python scripts/core/NB02_meander_detection.py       # ~8–16 hrs (4 sites)
python scripts/core/NB02_patch_width.py              # <5 min
python scripts/core/NB03_speed_eke_trends.py         # ~1–2 hrs
python scripts/core/NB04_argo_temperature.py         # ~10 min
python scripts/core/NB05a_era5_download.py           # ~30 min (316 MB download)
python scripts/core/NB05_era5_wind.py                # ~5 min

# === Figures ===
python figures/NB06_fig1.py                          # or NB06_figures.py for all
python figures/NB06_fig2.py
python figures/NB06_fig3.py
python figures/NB06_fig4.py

# === Supporting Information ===
python scripts/si/run_threshold.py 15                # Repeat for 25,30,35,40
python scripts/si/patch_width_all_thresholds.py
python scripts/si/compute_sensitivity_trends.py
python scripts/si/NB10_resolution_metric_comparison.py
python scripts/si/NB11_domain_map.py
python scripts/utils/combine_cp_bathymetry.py        # Pre-requisite for Fig 1/S3
```

## Methods Summary

### Meander Detection (NB02)
Adapted from the MATLAB codebase of [Liu et al. (2024)](https://doi.org/10.1029/2023JC019876): daily ADT gradient fields → 20% relative threshold → 4-month occurrence aggregation → peak extraction with half-peak-height width.

### Trend Analysis (NB03)
Sen's slope estimator with autocorrelation-adaptive Mann-Kendall significance testing. EKE computed as `spatial_mean(0.5 × (u'² + v'²))` with per-grid-point anomalies.

### Supporting Information
Threshold sensitivity (15–40%), resolution/metric comparison at CP and PAR (4 conditions: published, 0.25° zero-crossing, 0.125° zero-crossing, 0.125° half-peak-height), and domain definition maps.

## Related Publications

1. **Liu, X.**, Meyer, A., Langlais, C., & Lenton, A. (2024). Characteristics and trends of the Campbell Plateau meander in the Southern Ocean (1993–2020). *J. Geophys. Res.: Oceans*, 129, e2023JC019876.
2. **Liu, X.**, Yang, C., & Chen, Y. (2026). Standing meanders of the Antarctic Circumpolar Current: Evidence for ridge-controlled eddy saturation. *J. Geophys. Res.: Oceans*, 131, e2025JC023527.

## Citation

```bibtex
@article{Liu2026_GRL,
  author  = {Liu, Xinlong},
  title   = {Circumpolar Satellite Evidence for Topographically-Modulated
             Multi-Decadal Evolution of {Southern Ocean} Standing Meanders},
  journal = {Geophysical Research Letters},
  year    = {2026},
  note    = {In preparation}
}
```

## Author

**Xinlong Liu** — PhD Candidate, Institute for Marine and Antarctic Studies (IMAS), University of Tasmania

## License

MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgements

Satellite altimetry: EU Copernicus Marine Service (CMEMS). ERA5 reanalysis: Copernicus Climate Change Service (C3S). Argo floats: Roemmich-Gilson climatology (Scripps). Bathymetry: GMRT synthesis. Computations: NCI Gadi (project `gv90`), supported by NCRIS.
