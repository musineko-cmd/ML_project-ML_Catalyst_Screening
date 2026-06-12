"""HER-vs-OER comparison analysis."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def best_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics.sort_values("mae_mean").groupby(["reaction", "target"], as_index=False).first()


def read_top_importances(metrics_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in metrics_dir.glob("feature_importance_*.csv"):
        parts = path.stem.replace("feature_importance_", "").split("_")
        if len(parts) < 3:
            continue
        reaction = parts[0]
        model = parts[-1]
        target = "_".join(parts[1:-1])
        df = pd.read_csv(path).head(10)
        df["reaction"] = reaction
        df["target"] = target
        df["model"] = model
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def comparison_chart(best: pd.DataFrame, config: dict[str, Any]) -> None:
    labels = [f"{r}.{t}" for r, t in zip(best["reaction"], best["target"])]
    x = range(len(best))
    fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
    ax1.bar(x, best["mae_mean"], color="#557aa3", alpha=0.85, label="MAE")
    ax1.set_ylabel("MAE")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, rotation=35, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(list(x), best["r2_mean"], color="#b44b3f", marker="o", label="R2")
    ax2.set_ylabel("R2")
    ax1.set_title("Best cross-validated model by target")
    fig.tight_layout()
    fig.savefig(Path(config["paths"]["figures_dir"]) / "her_oer_comparison_metrics.png", dpi=300)
    plt.close(fig)


def top_feature_text(importances: pd.DataFrame, reaction: str) -> str:
    if importances.empty:
        return "No tree-model feature importance files were found."
    sub = importances[importances["reaction"] == reaction]
    if sub.empty:
        return "No importances available."
    summary = sub.groupby("feature")["importance"].mean().sort_values(ascending=False).head(8)
    return ", ".join(f"{feat} ({val:.3f})" for feat, val in summary.items())


def compare(config_path: str | Path = DEFAULT_CONFIG) -> str:
    config = load_config(config_path)
    metrics_dir = Path(config["paths"]["metrics_dir"])
    rankings_dir = Path(config["paths"]["rankings_dir"])
    metrics = pd.read_csv(metrics_dir / "cv_scores.csv")
    best = best_metrics(metrics)
    comparison_chart(best, config)

    her_best = best[best["reaction"] == "her"].iloc[0]
    oer_best = best[best["reaction"] == "oer"]
    oer_mae = float(oer_best["mae_mean"].mean())
    oer_r2 = float(oer_best["r2_mean"].mean())
    if float(her_best["r2_mean"]) > oer_r2:
        difficulty = (
            "HER was easier under this feature set, consistent with its single-descriptor target "
            "and less compound error propagation than the OER free-energy ladder"
        )
    else:
        difficulty = (
            "OER was easier in this bounded real-data run by average R2, likely because the complete "
            "OER triplets come from a smaller and more internally consistent subset of Catalysis-Hub, "
            "whereas the HER cache spans broader chemistry and duplicated site environments"
        )

    her_top = pd.read_csv(rankings_dir / "her_top.csv")
    oer_top = pd.read_csv(rankings_dir / "oer_top.csv")
    her_ids = set(her_top["candidate_id"].astype(str))
    oer_ids = set(oer_top["candidate_id"].astype(str))
    exact_overlap = sorted(her_ids & oer_ids)
    her_formulas = set(her_top["formula"].astype(str))
    oer_formulas = set(oer_top["formula"].astype(str))
    formula_overlap = sorted(her_formulas & oer_formulas)
    importances = read_top_importances(metrics_dir)

    fallback_note = ""
    if "fallback_used" in oer_top.columns and oer_top["fallback_used"].astype(str).str.lower().eq("true").any():
        fallback_note = "\n\nWARNING: OER screening used the documented synthetic fallback because complete Catalysis-Hub OH/O/OOH groups were insufficient in the bounded API pull."

    text = f"""# HER vs OER Comparison Summary

## Prediction Difficulty

- Best HER model for `dG_H`: {her_best['model']} with MAE={her_best['mae_mean']:.3f}±{her_best['mae_std']:.3f}, RMSE={her_best['rmse_mean']:.3f}±{her_best['rmse_std']:.3f}, R2={her_best['r2_mean']:.3f}±{her_best['r2_std']:.3f}.
- Mean best OER target performance: MAE={oer_mae:.3f}, R2={oer_r2:.3f} across `dG_OH`, `dG_O`, `dG_OOH`, and `overpotential`.
- Interpretation: {difficulty}.

## Feature Importance

- HER top tree-model features: {top_feature_text(importances, 'her')}.
- OER top tree-model features: {top_feature_text(importances, 'oer')}.

## Screening Overlap

- Top-{len(her_top)} exact candidate overlap: {len(exact_overlap)}.
- Top-{len(her_top)} formula overlap: {len(formula_overlap)}.
- Overlap list: {', '.join(formula_overlap[:10]) if formula_overlap else 'none'}.
- Bifunctional implication: little or no overlap suggests that optimizing HER near Delta G_H*=0 and minimizing OER overpotential are distinct objectives in this bounded dataset, so a bifunctional search should use multi-objective ranking rather than a single reaction proxy.{fallback_note}
"""
    out_path = Path(config["paths"]["comparison_summary"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"Saved comparison summary to {out_path}")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare HER and OER modeling/screening outcomes.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    compare(args.config)


if __name__ == "__main__":
    main()
