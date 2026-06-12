# Project Summary: HER/OER Catalyst Screening

## Purpose

This project builds a lightweight CPU-only tabular machine-learning pipeline for HER/OER catalyst screening. The goal is to predict adsorption-energy descriptors from catalyst composition/surface features, rank candidates with volcano-plot criteria, and compare HER vs OER behavior under one modeling framework.

## Data

The final run uses real Catalysis-Hub data only.

- HER raw data: 1200 Catalysis-Hub reaction rows
- OER raw data: 12362 Catalysis-Hub adsorbate rows
- Processed HER data: 937 rows
- Processed OER data: 94 complete OH*/O*/OOH* candidate groups
- Synthetic fallback: not used in the final reported results

Raw data are cached in `data/raw/`, and processed feature matrices are saved in `data/processed/`.

## Inputs and Targets

Model inputs are numeric features generated from formula, surface, facet, and limited site information.

Main feature sources:

- matminer Magpie composition descriptors
- elemental statistics such as electronegativity, radius, atomic number, group, period, valence proxy, d-electron proxy
- simple site/facet descriptors such as coordination-number proxy and facet digit sum

Prediction targets:

- HER: `dG_H`
- OER: `dG_OH`, `dG_O`, `dG_OOH`, and derived `overpotential`

OER overpotential is computed from the four-step free-energy ladder:

```text
DeltaG1 = dG_OH
DeltaG2 = dG_O - dG_OH
DeltaG3 = dG_OOH - dG_O
DeltaG4 = 4.92 - dG_OOH
```

## Pipeline

Run the full workflow with:

```bash
python run_all.py
```

Pipeline stages:

1. `src/data_acquisition.py`: query and cache Catalysis-Hub data
2. `src/featurize.py`: convert raw rows to fixed numeric feature matrices
3. `src/eda.py`: generate histograms, correlation heatmaps, and PCA plots
4. `src/models.py`: train/evaluate supervised regression models
5. `src/screening.py`: rank candidates and generate HER/OER volcano plots
6. `src/compare.py`: compare HER vs OER performance, feature importance, and top-candidate overlap

## Models

The project evaluates five supervised regression models with 5-fold cross-validation:

- LinearRegression
- Ridge
- RandomForestRegressor
- XGBRegressor
- MLPRegressor

Best models in the final real-data run:

- HER `dG_H`: RandomForestRegressor
- OER `dG_OH`: XGBRegressor
- OER `dG_O`: XGBRegressor
- OER `dG_OOH`: XGBRegressor
- OER `overpotential`: XGBRegressor

## Final Screening Results

### Updated HER-only run

After narrowing the project to HER, an expanded strict HER dataset was exported and used for improved modeling.

- Broad Hstar export: `data/raw/her_reactions_full.csv`, 36284 rows from a 150000-reaction bounded Catalysis-Hub scan
- Strict HER-only export: `data/raw/her_reactions_full_strict.csv`, 14731 rows
- Improved processed HER-only matrix: `data/processed/her_only_features.csv`, 9481 rows
- Key cleaning changes: clean H2/H* reactions only, per-H energy normalization by Hstar coefficient, target window `[-2.5, 2.5] eV`, candidate/site median aggregation, removal of high-variance duplicate labels
- Best HER-only model: ExtraTreesRegressor by MAE
- HER-only CV performance: MAE `0.260 ± 0.007`, RMSE `0.475 ± 0.014`, R2 `0.472 ± 0.021`
- Best R2 model: XGBRegressor with R2 about `0.497`

Updated HER-only top candidate:

- `Au6Fe6|AuFe|101`
- Formula: `Au6Fe6`
- Predicted `dG_H`: approximately `-0.00009 eV`

Updated HER-only outputs:

- Metrics: `results/metrics/her_only_cv_scores.csv`
- Predictions: `results/metrics/her_only_predictions.csv`
- Ranking: `results/rankings/her_only_top.csv`
- Figures: `results/figures/her_only_parity.png`, `results/figures/her_only_volcano.png`

## Rule-Based Candidate Generation

After training the HER-only surrogate model, a rule-based candidate space was generated for DFT validation prioritization.

Candidate definition includes composition, surface, facet, and adsorption site:

```text
candidate = formula + surface + facet + site
```

Generated candidate space:

- Elements: `Ag, Au, Pt, Pd, Ir, Rh, Ru, Ni, Co, Fe, Cu, Mo, W, Mn, Cr, Zn, Al, Ga, In, Sn`
- Binary alloy ratios: `A3B`, `AB`, `AB3`
- Facets: `111`, `100`, `101`
- Sites: `top`, `bridge`, `hollow`
- Total generated candidates: 5130
- Exact training-candidate novelty-filtered candidates: 4593

Ranking method:

```text
mu = predicted dG_H from the ExtraTrees HER surrogate
sigma = tree-ensemble prediction standard deviation
priority_score = |mu| + 0.5 * sigma
```

Lower `priority_score` means higher priority for future DFT validation.

Top exact-candidate-novel generated candidates:

1. `Ag3Cu9|AgCu3|111`, hollow site, predicted `dG_H ≈ 0.0099 eV`, score `0.0143`
2. `Au3Cu9|AuCu3|111`, hollow site, predicted `dG_H ≈ -0.0005 eV`, score `0.0198`
3. `Pt3Cu9|PtCu3|111`, bridge site, predicted `dG_H ≈ 0.0065 eV`, score `0.0203`

Top formula-novel generated candidates:

1. `Pt3Cu9|PtCu3|111`, bridge site, predicted `dG_H ≈ 0.0065 eV`, score `0.0203`
2. `W3Cr9|WCr3|111`, top site, predicted `dG_H ≈ 0.0103 eV`, score `0.0243`
3. `Pt9Mo3|Pt3Mo|111`, hollow site, predicted `dG_H ≈ -0.0231 eV`, score `0.0259`

Candidate-generation outputs:

- Generator script: `src/generate_candidates.py`
- All generated candidates: `results/rankings/generated_her_candidates.csv`
- DFT priority list: `results/rankings/her_dft_priority_list.csv`
- Formula-novel priority list: `results/rankings/her_dft_priority_list_formula_novel.csv`
- Uncertainty plot: `results/figures/generated_her_uncertainty.png`

### Original HER/OER comparison run

HER ranking criterion: smaller `|pred_dG_H|` is better.

Top HER candidate:

- `Ag9Ru3|Ag3Ru|111`
- Formula: `Ag9Ru3`
- Predicted `dG_H`: approximately `-0.005 eV`

OER ranking criterion: lower `pred_overpotential` is better.

Top OER candidate:

- `IrC26N4|IrN4C26-Δz=relaxed-M|001`
- Formula: `IrC26N4`
- Predicted overpotential: approximately `0.605 V`

Top-25 HER/OER overlap:

- Exact candidate overlap: 0
- Formula overlap: 0

Interpretation: HER and OER optimize different objectives in this bounded dataset. Bifunctional catalyst search should use multi-objective ranking rather than a single reaction score.

## Key Output Locations

- Raw data: `data/raw/`
- Processed feature matrices: `data/processed/`
- Figures: `results/figures/`
- Cross-validation scores: `results/metrics/cv_scores.csv`
- Model predictions: `results/metrics/predictions_her.csv`, `results/metrics/predictions_oer.csv`
- Feature importances: `results/metrics/feature_importance_*.csv`
- Rankings: `results/rankings/her_top.csv`, `results/rankings/oer_top.csv`
- Comparison summary: `results/comparison_summary.md`
- Report: `report/report.tex`, `report/report.pdf`
- Video outline: `report/video_outline.md`

## Important Notes

- Feature conversion is fixed and rule-based in `src/featurize.py`; the models do not learn the feature transformation.
- Predictions are produced by supervised regression models trained on Catalysis-Hub DFT-derived labels.
- The workflow is bounded and CPU-only; it does not download large bulk datasets.
- The project is intended as a clean ML course pipeline, not as a claim of newly discovered state-of-the-art catalysts.
