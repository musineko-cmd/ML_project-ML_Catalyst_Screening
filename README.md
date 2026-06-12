# HER/OER Catalyst Screening with Tabular Machine Learning

A one-semester graduate ML course term project. The goal is **not** to discover
new state-of-the-art catalysts, but to build a clean, well-analyzed ML pipeline
that screens catalyst candidates for two electrochemical reactions and compares
how the two reactions behave under an identical modeling framework.

## 1. Problem Definition

Electrocatalyst performance for water splitting is governed by intermediate
**adsorption (free) energies**:

- **HER** (Hydrogen Evolution Reaction): activity is described by the hydrogen
  adsorption free energy `ΔG_H*`. The volcano-plot optimum is `ΔG_H* ≈ 0 eV`.
- **OER** (Oxygen Evolution Reaction): activity is described by the free energies
  of the `OH*`, `O*`, `OOH*` intermediates and the resulting **theoretical
  overpotential** `η`.

We train tabular regressors to predict these descriptors from elemental /
compositional / structural features, then use the predictions to **rank and
screen** candidate materials via volcano plots. We compare HER vs. OER on
prediction difficulty, feature importance, and screening outcomes.

## 2. Relation to Course Content

The pipeline deliberately exercises core ML topics from the course:

- Supervised learning (regression)
- Linear models, regularization (Ridge/Lasso)
- Ensemble methods (Random Forest, gradient boosting / XGBoost)
- A small neural network (MLP) as an additional comparison
- Dimensionality reduction (PCA) for feature-space visualization
- Optimization (gradient-based MLP training, hyperparameter search)
- Evaluation methodology (k-fold cross-validation, MAE / RMSE / R²)
- Feature importance / interpretability

## 3. Dataset

**Primary source: Catalysis-Hub** (https://www.catalysis-hub.org) via its
GraphQL API. It provides DFT-computed reaction energies for adsorption
intermediates that map directly onto HER and OER descriptors.

- HER target: `ΔG_H*` (H adsorption energy).
- OER targets: `ΔG_OH*`, `ΔG_O*`, `ΔG_OOH*` → derived `η`.

Featurization is done with **matminer** (composition-based featurizers:
ElementProperty / Magpie, plus oxidation-state and valence features) and simple
structural/site descriptors where available (coordination number, etc.).

> The pipeline must stay lightweight: query a bounded subset (target a few
> thousand to ~tens of thousands of reactions), cache it to disk as CSV, and
> never attempt to download large bulk archives (e.g. full OC20).

If the Catalysis-Hub API is unavailable or returns too little data, fall back to
a Materials-Project / matminer compositional dataset of oxides for OER and a
metal-surface dataset for HER, documenting the substitution clearly.

## 4. Methodology

1. **Data acquisition**: pull HER and OER reaction energies, cache to
   `data/raw/`.
2. **Cleaning**: deduplicate, drop incomplete reactions, handle outliers,
   unify units (eV).
3. **Featurization**: build a feature matrix per reaction/site with matminer +
   hand-crafted elemental descriptors (electronegativity, atomic radius,
   valence electron count, group/period, d-band proxy, oxidation state,
   coordination number) plus statistical aggregates (mean/std).
4. **EDA**: target distributions, correlations, PCA scatter.
5. **Modeling**: Linear/Ridge baseline → Random Forest, XGBoost, MLP.
   5-fold cross-validation; light hyperparameter search.
6. **Screening**:
   - HER volcano plot (`ΔG_H*` vs predicted activity) → rank toward `ΔG_H* ≈ 0`.
   - OER volcano plot (`ΔG_O* − ΔG_OH*` descriptor vs `η`) → rank toward
     minimum overpotential.
7. **Comparison analysis** (the originality highlight): which reaction is harder
   to predict and why; how feature importances differ; whether top HER and top
   OER candidates overlap (bifunctional-catalyst discussion).

## 5. Repository Structure

```
her_oer_screening/
├── README.md
├── PROMPT.md                 # agent instructions
├── requirements.txt
├── config.yaml               # dataset sizes, model params, seeds
├── data/
│   ├── raw/                  # cached API pulls (CSV)
│   └── processed/            # feature matrices
├── src/
│   ├── data_acquisition.py   # Catalysis-Hub GraphQL pulls + caching
│   ├── featurize.py          # matminer + custom descriptors
│   ├── eda.py                # distributions, correlations, PCA
│   ├── models.py             # train/eval Linear, RF, XGB, MLP + CV
│   ├── screening.py          # volcano plots, candidate ranking
│   └── compare.py            # HER vs OER comparison analysis
├── results/
│   ├── figures/              # all plots (png, 300 dpi)
│   ├── metrics/              # CV scores, importance tables (csv)
│   └── rankings/             # screened candidate lists (csv)
├── report/
│   └── report.tex            # IEEE double-column draft
└── run_all.py                # end-to-end driver
```

## 6. Deliverables

- Final report in **IEEE conference style** (double-column, Times New Roman
  10 pt, single spacing), ~6–10 pages excluding references. Sections: problem
  definition, related work, methodology, model architecture, experimental
  setup, results & analysis, limitations & future work. No large code dumps in
  the report.
- Source code (this repo).
- Experimental results (figures, metrics, rankings).
- A presentation-video script/outline.

## 7. Constraints & Timeline

- **Two-week budget.** Week 1: data + featurization + EDA + baseline & main
  models. Week 2: CV/tuning + HER-vs-OER comparison + volcano screening +
  IEEE report + video outline.
- CPU-friendly. No GPU required for the tabular models. Keep dataset bounded.
- Reproducible: fixed random seeds, `config.yaml`, `requirements.txt`.

## 8. Quick Start

```bash
pip install -r requirements.txt
python run_all.py            # runs acquisition → featurize → models → screening → compare
```

Outputs land in `results/`. Edit `config.yaml` to change dataset size or models.

## DFT Validation Artifacts

The repository intentionally excludes VASP calculation inputs and heavy/restricted
outputs such as `POTCAR`, `WAVECAR`, `CHGCAR`, `OUTCAR`, `POSCAR`, `CONTCAR`,
`INCAR`, `KPOINTS`, and `vasprun.xml` from `dft_runs*` directories.

For submission/reproducibility, only lightweight DFT evidence is tracked:

- `dft_case_metadata.csv`
- `dft_h_adsorption_summary.csv`
- `dft_h_adsorption_status_summary.csv`
- `dft_h_adsorption_converged_only.csv`
- `OSZICAR` files and small scheduler stdout logs where available

This keeps the GitHub repository focused on the ML pipeline, candidate screening
results, parsed DFT energies, and convergence/status logs without uploading full
VASP calculation directories.
