# Agent Prompt — Build the HER/OER Catalyst Screening Project

You are an autonomous coding agent. Build the **entire** project described in
`README.md` end to end, in this repository, with minimal further questions. If a
choice is ambiguous, pick the simplest reasonable option, document it in a
comment or in the report, and keep going. The human reviewer is a materials-science
researcher; favor correct, well-commented, reproducible code over cleverness.

## Mission

Build a tabular-ML pipeline that predicts HER and OER adsorption-energy
descriptors from elemental/compositional/structural features, screens catalyst
candidates via volcano plots, and **compares HER vs. OER** under one framework.
This is a graduate ML *course* project — demonstrate solid ML methodology, not
novel SOTA results.

## Hard Constraints

- **Stay lightweight / CPU-only.** No GPU. Bound every data pull (target a few
  thousand to ~tens of thousands of reactions max). NEVER download large bulk
  archives (e.g. full OC20, hundreds of GB).
- **Cache everything** pulled from the network to `data/raw/*.csv` and read from
  cache on reruns. The pipeline must run offline after the first successful pull.
- **Reproducible**: set seeds everywhere, centralize knobs in `config.yaml`,
  pin versions in `requirements.txt`.
- **Two-week scope.** Do not gold-plate. Working > fancy.

## Step-by-Step Tasks

### 0. Scaffold
Create the directory structure from `README.md §5`. Write `requirements.txt`
(numpy, pandas, scikit-learn, xgboost, matminer, pymatgen, ase, matplotlib,
seaborn, requests, pyyaml — pin to recent stable versions) and a `config.yaml`
holding: dataset row caps, random_seed=42, CV folds=5, model hyperparameter
grids, and output paths.

### 1. Data acquisition (`src/data_acquisition.py`)
- Query **Catalysis-Hub** via its GraphQL endpoint
  (`http://api.catalysis-hub.org/graphql`) for reaction energies.
- Pull **HER**-relevant reactions (hydrogen adsorption, `H*`) and **OER**-relevant
  reactions (`OH*`, `O*`, `OOH*`).
- Page through results with a row cap from `config.yaml`. Extract: chemical
  composition / formula, surface or facet if present, reaction string,
  reaction energy (eV), and any site info.
- Save to `data/raw/her_reactions.csv` and `data/raw/oer_reactions.csv`.
- **Robustness**: wrap network calls in retries + timeouts. If the API fails or
  returns too few rows, fall back to a documented matminer/Materials-Project
  compositional dataset (oxides for OER, metals for HER), and print a clear
  WARNING about the substitution. The pipeline must always produce a usable
  cached dataset.

### 2. Featurization (`src/featurize.py`)
- Parse formulas with pymatgen `Composition`.
- Use matminer `ElementProperty.from_preset("magpie")` plus oxidation-state /
  valence featurizers for compositional descriptors.
- Add hand-crafted active-site descriptors where data allows: electronegativity,
  atomic radius, valence-electron count, group, period, a d-band-center proxy,
  oxidation state, coordination number; include mean/std aggregates.
- Drop all-NaN / constant columns; impute remaining NaNs (median). Output feature
  matrices to `data/processed/her_features.csv` and `oer_features.csv` with the
  regression target column(s) included.
- Targets: HER → `dG_H`. OER → `dG_OH`, `dG_O`, `dG_OOH`, and a derived
  `overpotential` computed from the 4-step mechanism
  (`η = max(ΔG1..ΔG4)/e − 1.23 V`, using standard OER scaling-relation steps).
  Document the exact formula in code comments.

### 3. EDA (`src/eda.py`)
Target histograms, feature-target correlation heatmaps, and a 2-D **PCA** scatter
of the feature space colored by target. Save all figures to `results/figures/`
at 300 dpi.

### 4. Modeling (`src/models.py`)
For HER and for each OER target:
- Models: `LinearRegression`, `Ridge`, `RandomForestRegressor`,
  `XGBRegressor`, and a small `MLPRegressor`.
- **5-fold cross-validation**; report MAE, RMSE, R² (mean ± std).
- Light hyperparameter search (small grid from `config.yaml`) for RF and XGB.
- Save a tidy metrics table to `results/metrics/cv_scores.csv` and feature
  importances (RF/XGB) to `results/metrics/feature_importance_*.csv`.
- Produce parity (predicted-vs-true) plots per best model into
  `results/figures/`.

### 5. Screening (`src/screening.py`)
- **HER volcano**: plot activity proxy vs `ΔG_H*`, optimum at 0 eV; rank all
  candidates by `|ΔG_H*|` ascending. Save `results/rankings/her_top.csv`.
- **OER volcano**: plot vs the standard `ΔG_O* − ΔG_OH*` descriptor against
  predicted `η`; rank by lowest `η`. Save `results/rankings/oer_top.csv`.
- Save both volcano figures to `results/figures/`.

### 6. Comparison analysis (`src/compare.py`) — the originality highlight
- Compare HER vs OER prediction difficulty (which has higher R² / lower error
  and a hypothesis why).
- Compare top feature importances across the two reactions.
- Compute overlap between top-HER and top-OER candidate sets; discuss
  bifunctional-catalyst implications.
- Emit a short markdown summary `results/comparison_summary.md` plus a comparison
  bar chart.

### 7. Driver (`run_all.py`)
Run acquisition → featurize → eda → models → screening → compare in order, with
clear logging at each stage and a final printed summary of where outputs landed.
Must succeed on a clean checkout with only `pip install -r requirements.txt`.

### 8. Report (`report/report.tex`)
Draft an **IEEE conference-style, double-column** paper (Times New Roman 10 pt,
single spacing, ~6–10 pages excl. references) with sections: Abstract, I.
Introduction / Problem Definition, II. Related Work, III. Methodology, IV. Model
Architecture, V. Experimental Setup, VI. Results & Analysis, VII. Limitations &
Future Work, References. Pull in generated figures from `results/figures/` with
`\includegraphics`. **Do not paste large code blocks.** Insert numeric results
from the metrics CSVs (you may hardcode the produced numbers after the run, but
mark them clearly). Add a short presentation-video outline as
`report/video_outline.md`.

## Definition of Done

- `python run_all.py` completes on CPU and populates `results/` with figures,
  metrics, and rankings.
- HER and OER are both modeled, screened, and compared.
- `report/report.tex` compiles in spirit (valid IEEEtran structure) and
  references the generated figures.
- Code is commented, seeded, and reads dataset/model knobs from `config.yaml`.
- A `WARNING` is printed (not a crash) if any fallback data path was used.

## Working Style

- Build incrementally and run each module as you finish it; fix errors before
  moving on.
- Prefer standard, well-documented libraries over exotic ones.
- Keep functions small and commented. Add docstrings.
- When the API schema is uncertain, first issue a tiny probe query, inspect the
  returned JSON shape, then write the full extractor against the real shape.
- Commit early and often if version control is available.
