# Presentation Video Outline

1. Problem setup, 30 seconds
Explain HER and OER descriptors: HER targets Delta G_H near 0 eV; OER needs OH, O, OOH intermediates and overpotential.

2. Data and constraints, 45 seconds
Show bounded Catalysis-Hub GraphQL pull, CSV caching, CPU-only design, and the documented OER fallback warning due incomplete triplets.

3. Features, 45 seconds
Describe matminer Magpie composition features plus hand-crafted elemental/site descriptors, NaN handling, and OER overpotential formula.

4. Models, 60 seconds
List LinearRegression, Ridge, RandomForest, XGBoost, and MLP. Emphasize 5-fold CV, fixed seed, and light grid search.

5. Results, 90 seconds
Show `her_oer_comparison_metrics.png` and parity plots. State best HER RF MAE 0.938 and R2 0.554; OER mean best MAE 0.099 and R2 about -0.006 with fallback limitation.

6. Screening, 60 seconds
Show HER and OER volcano plots. State top HER candidate `Bi9Re3|Bi3Re|111`; top OER demonstration candidate `Ag4Pt3O6|synthetic|110`.

7. Comparison and conclusion, 60 seconds
Summarize different feature importances, no top-25 overlap, and the implication that bifunctional catalyst screening should be multi-objective.

8. Limitations, 30 seconds
Clarify that OER fallback is not a discovery dataset; future work needs complete Catalysis-Hub OER groups, explicit surface descriptors, and uncertainty-aware ranking.
