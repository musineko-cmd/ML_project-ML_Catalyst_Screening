"""Predict HER H adsorption energies for an external generated candidate CSV.

This module turns the one-off generated-candidate screening step into a
reproducible script. It uses the HER-only training features, featurizes the input
candidate CSV with the same matminer/hand-crafted pipeline, and ranks candidates
with a small model ensemble plus an uncertainty penalty.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from featurize import load_config
from generate_candidates import featurize_candidates
from her_only import FEATURE_PATH, feature_columns, model_zoo


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.yaml"
DEFAULT_INPUT = ROOT / "generated_candidates.csv"
DEFAULT_PREDICTIONS = ROOT / "results/metrics/generated_candidates_ensemble_predictions.csv"
DEFAULT_PRIORITY_TOP = ROOT / "results/rankings/generated_candidates_top10_ensemble_priority.csv"
DEFAULT_NEAR_ZERO_TOP = ROOT / "results/rankings/generated_candidates_top10_ensemble_near_zero.csv"


def normalize_facet(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if text.isdigit() and len(text) < 3:
        return text.zfill(3)
    return text


def extract_h_site(value: Any) -> str:
    if pd.isna(value):
        return "unknown"
    text = str(value)
    try:
        parsed = json.loads(text)
    except Exception:
        return text
    if isinstance(parsed, dict):
        return str(parsed.get("H") or next(iter(parsed.values()), "unknown"))
    return text


def normalize_generated_candidates(raw: pd.DataFrame) -> pd.DataFrame:
    required = {"chemicalComposition", "surfaceComposition", "facet", "sites", "candidate_key"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required generated candidate columns: {sorted(missing)}")
    return pd.DataFrame(
        {
            "candidate_id": raw["candidate_key"].astype(str),
            "formula": raw["chemicalComposition"].astype(str),
            "surface": raw["surfaceComposition"].astype(str),
            "facet": raw["facet"].map(normalize_facet),
            "site": raw["sites"].map(extract_h_site),
            "source": "generated_candidates.csv",
            "fallback_used": False,
        }
    )


def selected_models(seed: int, n_jobs: int, requested: list[str]) -> dict[str, Any]:
    zoo = model_zoo(seed, n_jobs)
    models = {name: zoo[name] for name in requested if name in zoo}
    missing = [name for name in requested if name not in zoo]
    if missing:
        print(f"WARNING: requested models unavailable and skipped: {missing}")
    if not models:
        raise RuntimeError("No requested models are available.")
    return models


def model_prediction_with_uncertainty(model: Any, x_cand: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(model.predict(x_cand), dtype=float)
    if hasattr(model, "estimators_"):
        x_array = x_cand.to_numpy()
        tree_preds = np.vstack([tree.predict(x_array) for tree in model.estimators_])
        return tree_preds.mean(axis=0), tree_preds.std(axis=0)
    return pred, np.zeros_like(pred)


def predict_generated_candidates(
    config_path: str | Path = DEFAULT_CONFIG,
    input_path: str | Path = DEFAULT_INPUT,
    predictions_path: str | Path = DEFAULT_PREDICTIONS,
    priority_top_path: str | Path = DEFAULT_PRIORITY_TOP,
    near_zero_top_path: str | Path = DEFAULT_NEAR_ZERO_TOP,
    top_features: int = 280,
    uncertainty_weight: float = 0.5,
    model_names: list[str] | None = None,
) -> pd.DataFrame:
    config = load_config(config_path)
    train = pd.read_csv(FEATURE_PATH)
    raw = pd.read_csv(input_path)
    candidates = normalize_generated_candidates(raw)

    feats = feature_columns(train, top_features)
    x_train = train[feats]
    y_train = train["dG_H"]
    x_cand = featurize_candidates(candidates, config, train, feats)

    seed = int(config["random_seed"])
    n_jobs = int(config["models"].get("n_jobs", 2))
    requested = model_names or ["ExtraTrees", "RandomForest", "XGBoost"]
    models = selected_models(seed, n_jobs, requested)

    model_means: dict[str, np.ndarray] = {}
    model_sigmas: dict[str, np.ndarray] = {}
    for name, model in models.items():
        print(f"Fitting {name} on {len(train)} HER samples with {len(feats)} features")
        fitted = model.fit(x_train, y_train)
        mu, sigma = model_prediction_with_uncertainty(fitted, x_cand)
        model_means[name] = mu
        model_sigmas[name] = sigma

    mean_matrix = np.vstack([model_means[name] for name in models])
    within_var = np.vstack([model_sigmas[name] ** 2 for name in models])
    ensemble_mu = mean_matrix.mean(axis=0)
    model_sigma = mean_matrix.std(axis=0)
    within_sigma = np.sqrt(within_var.mean(axis=0))
    ensemble_sigma = np.sqrt(model_sigma**2 + within_sigma**2)

    out = raw.copy()
    out.insert(0, "candidate_id", candidates["candidate_id"])
    out["formula"] = candidates["formula"]
    out["surface"] = candidates["surface"]
    out["normalized_facet"] = candidates["facet"]
    out["site"] = candidates["site"]
    for name in models:
        out[f"pred_dG_H_{name}"] = model_means[name]
        out[f"pred_dG_H_sigma_{name}"] = model_sigmas[name]
    out["pred_dG_H_mu"] = ensemble_mu
    out["pred_dG_H_sigma"] = ensemble_sigma
    out["pred_dG_H_model_sigma"] = model_sigma
    out["pred_dG_H_within_model_sigma"] = within_sigma
    out["abs_pred_dG_H"] = np.abs(ensemble_mu)
    out["priority_score"] = out["abs_pred_dG_H"] + uncertainty_weight * out["pred_dG_H_sigma"]
    out["uncertainty_weight"] = uncertainty_weight
    out["models"] = "+".join(models)
    out["feature_mode"] = "matminer_magpie_plus_handcrafted_surface_site"
    out["n_train_samples"] = len(train)
    out["n_features"] = len(feats)

    ranked = out.sort_values("priority_score", ascending=True).reset_index(drop=True)
    near_zero = out.sort_values("abs_pred_dG_H", ascending=True).reset_index(drop=True)

    predictions_path = Path(predictions_path)
    priority_top_path = Path(priority_top_path)
    near_zero_top_path = Path(near_zero_top_path)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    priority_top_path.parent.mkdir(parents=True, exist_ok=True)
    near_zero_top_path.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(predictions_path, index=False)
    ranked.head(10).to_csv(priority_top_path, index=False)
    near_zero.head(10).to_csv(near_zero_top_path, index=False)

    print(f"Saved ensemble predictions -> {predictions_path}")
    print(f"Saved uncertainty-aware top 10 -> {priority_top_path}")
    print(f"Saved near-zero top 10 -> {near_zero_top_path}")
    print(
        ranked.head(10)[
            ["candidate_id", "pred_dG_H_mu", "pred_dG_H_sigma", "abs_pred_dG_H", "priority_score"]
        ].to_string(index=False)
    )
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict generated HER candidates with an uncertainty-aware ensemble.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--priority-top", default=str(DEFAULT_PRIORITY_TOP))
    parser.add_argument("--near-zero-top", default=str(DEFAULT_NEAR_ZERO_TOP))
    parser.add_argument("--top-features", type=int, default=280)
    parser.add_argument("--uncertainty-weight", type=float, default=0.5)
    parser.add_argument("--models", nargs="+", default=["ExtraTrees", "RandomForest", "XGBoost"])
    args = parser.parse_args()
    predict_generated_candidates(
        config_path=args.config,
        input_path=args.input,
        predictions_path=args.predictions,
        priority_top_path=args.priority_top,
        near_zero_top_path=args.near_zero_top,
        top_features=args.top_features,
        uncertainty_weight=args.uncertainty_weight,
        model_names=args.models,
    )


if __name__ == "__main__":
    main()
