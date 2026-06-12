# HER vs OER Comparison Summary

## Prediction Difficulty

- Best HER model for `dG_H`: RandomForest with MAE=1.108±0.086, RMSE=1.514±0.125, R2=0.373±0.046.
- Mean best OER target performance: MAE=0.539, R2=0.561 across `dG_OH`, `dG_O`, `dG_OOH`, and `overpotential`.
- Interpretation: OER was easier in this bounded real-data run by average R2, likely because the complete OER triplets come from a smaller and more internally consistent subset of Catalysis-Hub, whereas the HER cache spans broader chemistry and duplicated site environments.

## Feature Importance

- HER top tree-model features: matminer_MagpieData mode MendeleevNumber (0.210), matminer_MagpieData mean MeltingT (0.122), matminer_MagpieData mean MendeleevNumber (0.108), matminer_MagpieData mode Column (0.061), radius_max (0.042), matminer_MagpieData maximum CovalentRadius (0.030), matminer_MagpieData maximum MeltingT (0.028), matminer_MagpieData mean GSvolume_pa (0.027).
- OER top tree-model features: matminer_MagpieData maximum CovalentRadius (0.294), group_mean (0.116), matminer_MagpieData range NValence (0.107), matminer_MagpieData range NdUnfilled (0.099), matminer_MagpieData maximum NUnfilled (0.074), d_max (0.073), matminer_MagpieData avg_dev AtomicWeight (0.071), matminer_MagpieData minimum MendeleevNumber (0.068).

## Screening Overlap

- Top-25 exact candidate overlap: 0.
- Top-25 formula overlap: 0.
- Overlap list: none.
- Bifunctional implication: little or no overlap suggests that optimizing HER near Delta G_H*=0 and minimizing OER overpotential are distinct objectives in this bounded dataset, so a bifunctional search should use multi-objective ranking rather than a single reaction proxy.
