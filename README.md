# Multiple Linear Regression Outperforms Machine Learning for Monthly Temperature Forecasting Across the Peruvian Altiplano

**Status:** Manuscript under preparation
**Institution:** Universidad Nacional del Altiplano de Puno, Peru
**License:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-15803d?style=flat-square)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776ab?style=flat-square)](https://www.python.org/)
[![Data: NASA POWER](https://img.shields.io/badge/Data-NASA_POWER-0b3d91?style=flat-square)](https://power.larc.nasa.gov/)
[![Stations](https://img.shields.io/badge/Series-11_independent-64748b?style=flat-square)](#replication)
[![Reproducible](https://img.shields.io/badge/Pipeline-Fully_reproducible-1d4ed8?style=flat-square)](#reproducing-the-analysis)

---

## Authors

| Name | Institution |
|---|---|
| Leonel Coyla Idme | Universidad Nacional del Altiplano de Puno, Peru |
| Vidman Ruiz Roque Mamani | Universidad Nacional del Altiplano de Puno, Peru |
| Smit Alexander Suni Morales | Universidad Nacional del Altiplano de Puno, Peru |
| Keysi Salcca Lagar | Universidad Nacional del Altiplano de Puno, Peru |
| Alex Arias Ramírez | Universidad Nacional del Altiplano de Puno, Peru |
| Antony Jhonatan Flores Nina | Universidad Nacional del Altiplano de Puno, Peru |
| Richar Andre Vilca Solorzano | Universidad Nacional del Altiplano de Puno, Peru |

**Faculty:** Ingenieria Estadistica e Informatica — Universidad Nacional del Altiplano (UNA), Puno, Peru

---

## Overview

Monthly temperature forecasts guide planting calendars, irrigation scheduling and frost-protection decisions in the Peruvian Altiplano, where a single frost outside the expected window can destroy a season of quinoa or potato production. Studies in the region routinely report coefficients of determination near 0.90 for machine learning models — but in a regime whose seasonal cycle spans 4.68 °C while within-month variability averages 0.71 °C, **a model that reproduces only the annual cycle already achieves that number**.

This study asks a narrower and harder question: *does machine learning beat the climatological reference that operational meteorology has used for a century?*

The answer is yes, but modestly — and the simplest model wins.

**Key finding:** multiple linear regression matches or outperforms a tuned multilayer perceptron and a tuned random forest for one-month-ahead forecasting, and the apparent advantage of the neural model turns out to be an artefact of random initialisation.

---

## Research Contributions

1. **Skill measured against the correct reference.** Improvement is reported as a mean square error skill score relative to monthly climatology, not as an unreferenced R².
2. **The neural advantage does not replicate.** Across 50 random seeds the perceptron averages 0.495 °C, not the 0.415 °C of a single fit; only 4 of 50 initialisations beat the deterministic linear model.
3. **Replication across 11 independent series.** The protocol is repeated on eleven Altiplano locations and 792 test months, with linear regression best at 9 of 11 sites.
4. **The sample-size explanation, tested rather than assumed.** Learning curves extended to 2,475 training months — eighteen times the single-station record — show the machine learning curves flattening *above* the linear model without ever crossing it.
5. **Auditable data handling.** WMO-No. 1203 completeness rule, a regression-calibrated reconstruction whose bias is measured (not assumed), and a permutation test confirming the absence of leakage.

---

## Results

### Puno Principal Climatological Station (49 test months)

| Model | MAE (°C) | RMSE (°C) | R² | Skill vs. climatology |
|---|---|---|---|---|
| Persistence | 0.866 | 1.059 | 0.520 | −0.958 |
| Climatology *(reference)* | 0.597 | 0.757 | 0.755 | 0.000 |
| **Multiple linear regression** | **0.421** | **0.532** | **0.879** | **+0.505** |
| Random forest | 0.476 | 0.629 | 0.831 | +0.310 |
| Multilayer perceptron *(single seed)* | 0.415 | 0.541 | 0.875 | +0.489 |

Persistence performs **worse than doing nothing**: in a strongly seasonal regime, carrying the previous month forward is worse than ignoring it.

### Seed sensitivity — why the single-seed result is misleading

| Model | Single fit | Mean over 50 seeds | SD | Range | Seeds beating MLR |
|---|---|---|---|---|---|
| Multiple linear regression | 0.421 | *deterministic* | — | — | — |
| Random forest | 0.476 | 0.476 | 0.006 | 0.464–0.486 | **0 / 50** |
| Multilayer perceptron | 0.415 | **0.495** | 0.048 | 0.407–0.647 | **4 / 50** |

The reported 0.415 °C sits at the **6th percentile** of the perceptron's own sampling distribution. An ensemble of all fifty networks reaches 0.4209 °C — statistically identical to the 0.4208 °C of a twelve-parameter closed-form regression.

### Replication

Eleven independent Altiplano series, 792 pooled test months:

| | Climatology | MLR | RF | MLP |
|---|---|---|---|---|
| Pooled MAE (°C) | 0.640 | **0.438** | 0.474 | 0.453 |
| Sites where best | — | **9 / 11** | 0 / 11 | 2 / 11 |

### Learning curves

| Pooled training months | MLR | RF | MLP |
|---|---|---|---|
| 100 | 0.547 | 0.580 | 0.600 |
| 700 | **0.424** | 0.499 | 0.456 |
| 2,475 | **0.438** | 0.477 | 0.467 |

Eighteen times the data does not change the ordering.

---

## Data

| Source | Coverage | Role |
|---|---|---|
| SENAMHI, Puno Principal Climatological Station | 7,279 daily records, 2003–2024 | Primary analysis — ground truth |
| [NASA POWER](https://power.larc.nasa.gov/) (MERRA-2) | 13 locations × 9,132 daily records, 2000–2024 | Replication — open data, no registration |

**Two data problems found and documented, not hidden:**

- The workbook supplied for this study contained a second sheet labelled as a different station. Cell-by-cell comparison across 65,538 cells found **exactly one difference**: the station label. It was excluded.
- Of the 13 NASA POWER locations, two pairs return **numerically identical series** because each pair falls inside the same MERRA-2 grid cell (~0.5° × 0.625°). Eleven independent series remain.
- NASA POWER is **2.577 °C colder** than the station record at Puno (paired *t* = −178.3), consistent with the 4,105 m elevation MERRA-2 assigns to a cell whose station sits at 3,825 m. Monthly correlation is nonetheless 0.955, so the reanalysis tracks variability well but cannot serve as ground truth.

---

## Repository Structure

```
monthly-temperature-forecasting-altiplano/
│
├── data/
│   └── senamhi_puno_2003_2024.xlsx      # Primary station record
│
├── analysis/
│   ├── config.py                        # Paths, constants, WMO thresholds
│   ├── 01_load_aggregate.py             # Load, audit, detect duplicate sheets
│   ├── 02_calibrate_and_build.py        # Calibrated reconstruction + monthly series
│   ├── 03_models.py                     # Models, baselines, Diebold-Mariano, bootstrap
│   ├── 04_figures.py                    # Figures 1-5
│   ├── 05_tables.py                     # Tables 1-6
│   ├── 06_extended_analysis.py          # Ablation, seasonality, residuals, learning curve
│   ├── 07_figures_extended.py           # Figures 6-7
│   ├── 08_sanity_checks.py              # Leakage test, split integrity, missing data
│   ├── 09_scaling_extrapolation.py      # Power-law fit and crossing-point bootstrap
│   ├── 10_seed_stability.py             # 50-seed replication of stochastic models
│   ├── 11_download_nasa_power.py        # NASA POWER retrieval (public API)
│   ├── 12_validate_nasa_power.py        # Independence check + validation vs. SENAMHI
│   ├── 13_multistation_replication.py   # 11-series replication, pooled learning curves
│   ├── 14_figures_replication.py        # Figures 8-9
│   │
│   ├── data/                            # Generated intermediates + NASA POWER CSVs
│   ├── results/                         # JSON reports and publication tables
│   └── figures/                         # Figures at 300 dpi (TIFF + PNG)
```

---

## Reproducing the Analysis

```bash
git clone https://github.com/Andre031222/monthly-temperature-forecasting-altiplano.git
cd monthly-temperature-forecasting-altiplano
pip install -r requirements.txt

cd analysis
python 01_load_aggregate.py          # audit and duplicate-sheet detection
python 02_calibrate_and_build.py     # calibration and monthly aggregation
python 03_models.py                  # models, baselines, significance tests
python 04_figures.py
python 05_tables.py
python 06_extended_analysis.py
python 07_figures_extended.py
python 08_sanity_checks.py           # leakage and integrity checks
python 09_scaling_extrapolation.py
python 10_seed_stability.py
python 11_download_nasa_power.py     # downloads 13 series from the public API
python 12_validate_nasa_power.py
python 13_multistation_replication.py
python 14_figures_replication.py
```

Scripts run in order; each writes its own JSON report to `analysis/results/`. Every random operation is seeded, so the numbers above reproduce exactly. Total runtime is roughly 25 minutes on a standard laptop, most of it in hyperparameter search.

**No API key or registration is required.** `11_download_nasa_power.py` uses the public NASA POWER endpoint directly.

---

## Methodological Notes

**Why blocked cross-validation.** Randomly shuffling a temporal series lets information from future months reach the fitted model. Training uses `TimeSeriesSplit` with five blocked folds, and the test set is consulted once, after hyperparameter selection closes.

**Why the midpoint estimator was calibrated.** Daily mean temperature is missing on 10.72% of records while the extremes are missing on under 2%. The naive estimator (Tmax + Tmin)/2 underestimates the observed mean by 0.687 °C (*t* = −65.21), because the recorded mean derives from synoptic-hour observations. A regression fitted **only on training years** removes the bias and recovers 606 daily values.

**Why a leakage test.** Shuffling the target 200 times and refitting yields a mean MAE of 1.402 °C against 0.421 °C for the real model, with **0 of 200** shuffled models coming close (*p* = 0.005).

---

## License

Code and documentation are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). NASA POWER data are in the public domain. The SENAMHI station record is redistributed for reproducibility; original rights belong to the Servicio Nacional de Meteorología e Hidrología del Perú.

---

## Acknowledgments

Instituto de Investigación and Vicerrectorado de Investigación, Universidad Nacional del Altiplano de Puno, through the Research Seedbeds programme. Climatological records provided by SENAMHI. Reanalysis data from the NASA Langley Research Center POWER Project.
