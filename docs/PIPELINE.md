# Pipeline Execution Guide

Complete guide for reproducing all analyses in the GRL manuscript and Supporting Information.

## Platform

- **HPC**: NCI Gadi (project `gv90`, username `xl1657`)
- **Python**: 3.11 via `/g/data/gv90/xl1657/venvs/cmems_py311/`
- **Access**: JupyterLab via ARE dashboard or PBS batch jobs (normal queue)

## Input Data

Primary input: `/g/data/gv90/xl1657/cmems_adt/cmems_so30S_19930101_20250802_adt_sla_ugos_vgos_batch.nc`

Contains daily `adt`, `sla`, `ugos`, `vgos` at 0.125° (90°S–30°S, 1993–2025).

## Script Inventory (21 files)

### Core Pipeline (6 scripts)

| Script | Purpose | Input | Output | Runtime |
|--------|---------|-------|--------|---------|
| `NB02_meander_detection.py` | ADT gradient meander detection (4 sites) | CMEMS ADT | NetCDF + CSV per site | 8–16 hrs |
| `NB02_patch_width.py` | Half-peak-height width correction | NB02 NetCDFs | Patched NetCDF + CSV | <5 min |
| `NB03_speed_eke_trends.py` | Speed, EKE, Mann-Kendall trends | NB02 CSVs + ADT | Updated CSVs + trends CSV | 1–2 hrs |
| `NB04_argo_temperature.py` | Argo 500 dbar temperature analysis | Roemmich-Gilson Argo | Argo summary CSV | ~10 min |
| `NB05a_era5_download.py` | Download ERA5 wind from CDS | CDS API | ERA5 NetCDF (316 MB) | ~30 min |
| `NB05_era5_wind.py` | ERA5 wind trends per site | ERA5 NetCDF | Wind CSVs + trends | ~5 min |

### Supporting Information (6 scripts)

| Script | Purpose | SI Component |
|--------|---------|-------------|
| `NB09_threshold_sensitivity.py` | Full threshold sensitivity orchestrator | Figure S1, Table S1 |
| `run_threshold.py` | CLI wrapper: run NB02 at one threshold | (Helper for NB09) |
| `patch_width_all_thresholds.py` | Apply half-peak-height width to all thresholds | (Helper for NB09) |
| `compute_sensitivity_trends.py` | Mann-Kendall trends across all thresholds | `all_threshold_trends.csv` |
| `NB10_resolution_metric_comparison.py` | 4-condition resolution/metric comparison (CP, PAR) | Figure S2, Table S2 |
| `NB11_domain_map.py` | Domain definition map with bathymetry | Figure S3 |

### Figures (5 scripts)

| Script | Figure | Content |
|--------|--------|---------|
| `NB06_figures.py` | All 4 figures | Combined script |
| `NB06_fig1.py` | Figure 1 | Circumpolar map + 4 zoom panels with decadal meander positions |
| `NB06_fig2.py` | Figure 2 | 4-panel trend time series (position, width, speed, EKE) |
| `NB06_fig3.py` | Figure 3 | Cross-site bar comparison with significance markers |
| `NB06_fig4.py` | Figure 4 | Forcing context: ERA5 wind trends + Argo temperature |

### Utilities (1 script)

| Script | Purpose |
|--------|---------|
| `combine_cp_bathymetry.py` | Combine CP East + West GMRT tiles across the dateline → single NetCDF |

### Notebooks (3 files)

Interactive Jupyter versions of NB02, NB03, and NB06 for exploratory analysis.

## Execution Order

### Phase 1: Core Analysis

```bash
source /g/data/gv90/xl1657/venvs/cmems_py311/bin/activate

# Step 1: Meander detection
python scripts/core/NB02_meander_detection.py

# Step 1b: Width correction
python scripts/core/NB02_patch_width.py

# Step 2: Speed, EKE, trends
python scripts/core/NB03_speed_eke_trends.py

# Step 3: Forcing context
python scripts/core/NB04_argo_temperature.py
python scripts/core/NB05a_era5_download.py
python scripts/core/NB05_era5_wind.py
```

### Phase 2: Figures

```bash
# Pre-requisite for Figure 1
python scripts/utils/combine_cp_bathymetry.py

# All figures
python figures/NB06_fig1.py
python figures/NB06_fig2.py
python figures/NB06_fig3.py
python figures/NB06_fig4.py
```

### Phase 3: Supporting Information

```bash
# Threshold sensitivity (computationally intensive — use PBS)
for thresh in 15 25 30 35 40; do
    python scripts/si/run_threshold.py $thresh
done
python scripts/si/patch_width_all_thresholds.py
python scripts/si/compute_sensitivity_trends.py

# Resolution/metric comparison (CP and PAR only)
python scripts/si/NB10_resolution_metric_comparison.py

# Domain map
python scripts/si/NB11_domain_map.py
```

## Key Design Decisions

**EKE correction**: `spatial_mean(0.5 × (u'² + v'²))` with per-grid-point anomalies relative to monthly climatology. The original version incorrectly used spatially averaged velocity anomalies (~1000× too small).

**Width metric**: Half-peak-height (matching MATLAB `findpeaks('WidthReference','halfheight')`), applied via `NB02_patch_width.py` after initial zero-crossing detection.

**Dateline handling**: CP site (150°E to 150°W) reads ADT in two longitude slices and concatenates. GMRT bathymetry similarly combined via `combine_cp_bathymetry.py`.

**Autocorrelation-adaptive MK**: Modified Mann-Kendall (Hamed & Rao 1998) when |lag-1 ACF| > 0.1; original otherwise.

## Trend Summary (Corrected Pipeline)

| Site | Position (°/dec) | Width (km/dec) | Speed (m s⁻¹/dec) | EKE (m² s⁻²/dec) |
|------|:-:|:-:|:-:|:-:|
| CP   | +0.084 (ns) | **−3.78** ** | **+7.58×10⁻³** *** | **+9.47×10⁻⁴** *** |
| PAR  | +0.018 (ns) | **+4.13** *** | **+7.92×10⁻³** *** | **+8.72×10⁻⁴** *** |
| SEIR | −0.060 (ns) | **−4.14** ** | **+6.43×10⁻³** *** | **+1.03×10⁻³** *** |
| SWIR | **−0.109** * | −0.96 (ns) | **+2.85×10⁻³** *** | **+3.13×10⁻⁴** * |

Bold = significant. \* *p* < 0.05, \*\* *p* < 0.01, \*\*\* *p* < 0.001. 12 of 16 significant.
