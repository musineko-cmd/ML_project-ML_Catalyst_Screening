"""Rule-based HER candidate generation and uncertainty-aware ranking."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from featurize import load_config
from her_only import FEATURE_PATH, RAW_PATH, feature_columns, her_feature_frame, model_zoo


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.yaml"
GENERATED_PATH = ROOT / "results/rankings/generated_her_candidates.csv"
PRIORITY_PATH = ROOT / "results/rankings/her_dft_priority_list.csv"
FORMULA_NOVEL_PRIORITY_PATH = ROOT / "results/rankings/her_dft_priority_list_formula_novel.csv"
FIGURE_PATH = ROOT / "results/figures/generated_her_uncertainty.png"

ELEMENTS = [
    "Ag",
    "Au",
    "Pt",
    "Pd",
    "Ir",
    "Rh",
    "Ru",
    "Ni",
    "Co",
    "Fe",
    "Cu",
    "Mo",
    "W",
    "Mn",
    "Cr",
    "Zn",
    "Al",
    "Ga",
    "In",
    "Sn",
]
RATIOS = {
    "A3B": (9, 3, "{a}3{b}"),
    "AB": (6, 6, "{a}{b}"),
    "AB3": (3, 9, "{a}{b}3"),
}
FACETS = ["111", "100", "101"]
SITES = ["top", "bridge", "hollow"]


def _formula(a: str, b: str, ca: int, cb: int) -> str:
    return f"{a}{ca if ca != 1 else ''}{b}{cb if cb != 1 else ''}"


def generate_rule_based_candidates() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for a, b in combinations(ELEMENTS, 2):
        for ratio_name, (ca, cb, surface_template) in RATIOS.items():
            formula = _formula(a, b, ca, cb)
            surface = surface_template.format(a=a, b=b)
            for facet in FACETS:
                for site in SITES:
                    rows.append(
                        {
                            "candidate_id": f"{formula}|{surface}|{facet}",
                            "formula": formula,
                            "surface": surface,
                            "facet": facet,
                            "site": site,
                            "ratio": ratio_name,
                            "source": "rule_based_generated",
                            "fallback_used": False,
                        }
                    )
    return pd.DataFrame(rows)


def novelty_filter(candidates: pd.DataFrame) -> pd.DataFrame:
    raw = pd.read_csv(RAW_PATH)
    seen_candidate_ids = set(raw["candidate_id"].astype(str))
    seen_formulas = set(raw["formula"].astype(str))
    out = candidates.copy()
    out["candidate_in_training"] = out["candidate_id"].astype(str).isin(seen_candidate_ids)
    out["formula_in_training"] = out["formula"].astype(str).isin(seen_formulas)
    # Exact candidate novelty is the hard filter. Formula overlap is retained as
    # a diagnostic because familiar formulas can still represent new facets/sites.
    return out[~out["candidate_in_training"]].reset_index(drop=True)


def featurize_candidates(candidates: pd.DataFrame, config: dict[str, Any], train: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    cand_features = her_feature_frame(candidates, config).replace([np.inf, -np.inf], np.nan)

    train_medians = train.median(numeric_only=True)
    aligned_cols: dict[str, pd.Series] = {}
    for col in feats:
        if col in cand_features:
            series = pd.to_numeric(cand_features[col], errors="coerce")
        else:
            series = pd.Series(np.nan, index=candidates.index)
        fill_value = float(train_medians[col]) if col in train_medians else 0.0
        aligned_cols[col] = series.fillna(fill_value)
    return pd.DataFrame(aligned_cols, index=candidates.index)


def rank_candidates(config_path: str | Path = DEFAULT_CONFIG, top_features: int = 280, uncertainty_weight: float = 0.5) -> pd.DataFrame:
    config = load_config(config_path)
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(f"Missing {FEATURE_PATH}; run `python src/her_only.py` first.")

    train = pd.read_csv(FEATURE_PATH)
    feats = feature_columns(train, top_features)
    x_train = train[feats]
    y_train = train["dG_H"]

    candidates = novelty_filter(generate_rule_based_candidates())
    x_cand = featurize_candidates(candidates, config, train, feats)

    seed = int(config["random_seed"])
    n_jobs = int(config["models"].get("n_jobs", 2))
    model = model_zoo(seed, n_jobs)["ExtraTrees"]
    model.fit(x_train, y_train)

    x_cand_array = x_cand.to_numpy()
    tree_preds = np.vstack([tree.predict(x_cand_array) for tree in model.estimators_])
    out = candidates.copy()
    out["pred_dG_H_mu"] = tree_preds.mean(axis=0)
    out["pred_dG_H_sigma"] = tree_preds.std(axis=0)
    out["activity_distance"] = out["pred_dG_H_mu"].abs()
    out["priority_score"] = out["activity_distance"] + uncertainty_weight * out["pred_dG_H_sigma"]
    out["uncertainty_weight"] = uncertainty_weight
    out = out.sort_values("priority_score", ascending=True).reset_index(drop=True)

    GENERATED_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(GENERATED_PATH, index=False)
    out.head(100).to_csv(PRIORITY_PATH, index=False)
    out[~out["formula_in_training"]].head(100).to_csv(FORMULA_NOVEL_PRIORITY_PATH, index=False)
    _save_uncertainty_plot(out)
    print(f"Generated novel candidates: {len(out)} -> {GENERATED_PATH}")
    print(f"Saved DFT priority list: {PRIORITY_PATH}")
    print(f"Saved formula-novel DFT priority list: {FORMULA_NOVEL_PRIORITY_PATH}")
    print(f"Top generated candidate: {out.iloc[0]['candidate_id']} score={out.iloc[0]['priority_score']:.4f}")
    return out


def _save_uncertainty_plot(out: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.3, 4.1))
    sc = ax.scatter(
        out["pred_dG_H_mu"],
        out["pred_dG_H_sigma"],
        c=out["priority_score"],
        cmap="viridis_r",
        s=16,
        alpha=0.75,
    )
    ax.axvline(0.0, linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("Predicted Delta G_H* mean (eV)")
    ax.set_ylabel("Ensemble uncertainty sigma (eV)")
    ax.set_title("Generated HER candidates: activity vs uncertainty")
    fig.colorbar(sc, ax=ax, label="Priority score, lower is better")
    fig.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and rank rule-based HER candidates.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--top-features", type=int, default=280)
    parser.add_argument("--uncertainty-weight", type=float, default=0.5)
    args = parser.parse_args()
    rank_candidates(args.config, args.top_features, args.uncertainty_weight)


if __name__ == "__main__":
    main()
