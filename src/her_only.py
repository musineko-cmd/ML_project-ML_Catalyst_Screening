"""Improved HER-only filtering, featurization, modeling, and screening.

This module is intentionally separate from the HER/OER comparison pipeline. It
uses the larger strict HER export, normalizes multi-H reactions to per-H energies,
reduces duplicate-label noise by candidate/site aggregation, and evaluates a
slightly stronger tabular model set.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None

from featurize import ELEMENT_PROPS, build_feature_frame, clean_features, composition_features, load_config


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.yaml"
RAW_PATH = ROOT / "data/raw/her_reactions_full_strict.csv"
FEATURE_PATH = ROOT / "data/processed/her_only_features.csv"
METRICS_PATH = ROOT / "results/metrics/her_only_cv_scores.csv"
PRED_PATH = ROOT / "results/metrics/her_only_predictions.csv"
RANKING_PATH = ROOT / "results/rankings/her_only_top.csv"
PARITY_PATH = ROOT / "results/figures/her_only_parity.png"
VOLCANO_PATH = ROOT / "results/figures/her_only_volcano.png"

NON_FEATURE = {
    "candidate_id",
    "formula",
    "surface",
    "facet",
    "site",
    "source",
    "fallback_used",
    "dG_H",
    "label_std",
    "label_count",
}


def _loads(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value in (None, "") or (isinstance(value, float) and np.isnan(value)):
        return {}
    try:
        return json.loads(value)
    except Exception:
        return {}


def hstar_coeff(products: Any) -> float:
    coeff = _loads(products).get("Hstar", 1.0)
    try:
        return max(float(coeff), 1.0)
    except Exception:
        return 1.0


def strict_clean_her(raw: pd.DataFrame, target_window: tuple[float, float], max_label_std: float) -> pd.DataFrame:
    """Normalize, outlier-filter, and aggregate clean HER reactions.

    Catalysis-Hub reaction energies may correspond to multiple adsorbed H atoms
    in reactions such as H2 + 2* -> 2H*. For HER volcano screening the descriptor
    is per-H adsorption energy, so reactionEnergy is divided by the Hstar product
    coefficient before modeling.
    """

    df = raw.copy()
    h_decorated = (
        df["formula"].astype(str).map(_has_explicit_h_decoration)
        | df["surface"].astype(str).map(_has_explicit_h_decoration)
        | df["facet"].astype(str).str.contains(r"-H|H-covered|hydrogen", case=False, regex=True, na=False)
    )
    df = df[~h_decorated].copy()
    df["hstar_coeff"] = df["products"].map(hstar_coeff)
    df["dG_H"] = pd.to_numeric(df["reaction_energy"], errors="coerce") / df["hstar_coeff"]
    df = df.dropna(subset=["candidate_id", "formula", "surface", "facet", "dG_H"])
    lo, hi = target_window
    df = df[df["dG_H"].between(lo, hi)]

    keys = ["candidate_id", "formula", "surface", "facet", "site", "source", "fallback_used"]
    grouped = df.groupby(keys, dropna=False)["dG_H"].agg(["median", "std", "count"]).reset_index()
    grouped = grouped.rename(columns={"median": "dG_H", "std": "label_std", "count": "label_count"})
    grouped["label_std"] = grouped["label_std"].fillna(0.0)
    grouped = grouped[grouped["label_std"] <= max_label_std].reset_index(drop=True)
    return grouped


def _has_explicit_h_decoration(text: str) -> bool:
    """Detect pre-hydrogenated catalyst strings such as Rh36H4 or 111-H.

    These are excluded for the HER-only screening run because the generated
    candidates represent clean catalyst surfaces before the target H adsorption.
    """

    return bool(re.search(r"(^|[^a-z])H([0-9]|$)", text)) or "-H" in text


def _surface_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for surface in df["surface"]:
        feats = composition_features(surface)
        rows.append({f"surface_{k}": v for k, v in feats.items()})
    return pd.DataFrame(rows, index=df.index)


def _surface_base_label(surface: Any) -> str:
    return str(surface or "").split("-")[0].replace("+1%", "")


def _surface_dopant_label(surface: Any) -> str:
    for token in str(surface or "").split("-")[1:]:
        if token in ELEMENT_PROPS:
            return token
    return ""


def _surface_label_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    props = ["Z", "en", "radius", "group", "period", "valence", "d", "ox", "d_band_proxy"]
    for surface in df["surface"]:
        text = str(surface or "")
        lower = text.lower()
        base_label = _surface_base_label(surface)
        dopant = _surface_dopant_label(surface)
        base_feats = composition_features(base_label)
        dopant_feats = composition_features(dopant) if dopant else {}
        row: dict[str, float] = {
            "surface_label_has_dopant": float(bool(dopant)),
            "surface_label_is_oxide": float("O" in base_label),
            "surface_label_has_lattice": float("lattice" in lower),
            "surface_label_has_bridge_tag": float("brg" in lower or "bridge" in lower),
        }
        pct = re.search(r"([0-9]+(?:\.[0-9]+)?)%", text)
        row["surface_label_perturbation_percent"] = float(pct.group(1)) if pct else 0.0
        site_num = re.search(r"[A-Za-z]+([0-9]+)$", text.split("-")[-1])
        row["surface_label_suffix_number"] = float(site_num.group(1)) if site_num else 0.0
        for prop in props:
            base_val = base_feats.get(f"{prop}_mean", base_feats.get(prop, np.nan))
            dopant_val = dopant_feats.get(f"{prop}_mean", dopant_feats.get(prop, np.nan))
            row[f"surface_base_{prop}"] = base_val
            row[f"surface_dopant_{prop}"] = dopant_val
            row[f"surface_dopant_minus_base_{prop}"] = dopant_val - base_val if pd.notna(dopant_val) and pd.notna(base_val) else np.nan
        rows.append(row)
    return pd.DataFrame(rows, index=df.index)


def _interaction_features(base: pd.DataFrame, surface: pd.DataFrame) -> pd.DataFrame:
    rows: dict[str, pd.Series] = {}
    comparable = [
        "oxygen_fraction",
        "Z_mean",
        "en_mean",
        "radius_mean",
        "group_mean",
        "period_mean",
        "valence_mean",
        "d_mean",
        "ox_mean",
        "d_band_proxy",
        "metal_atom_fraction",
    ]
    for col in comparable:
        surface_col = f"surface_{col}"
        if col not in base or surface_col not in surface:
            continue
        delta = pd.to_numeric(base[col], errors="coerce") - pd.to_numeric(surface[surface_col], errors="coerce")
        rows[f"delta_formula_surface_{col}"] = delta
        rows[f"abs_delta_formula_surface_{col}"] = delta.abs()
        if col.endswith("_mean") or col in {"d_band_proxy", "metal_atom_fraction"}:
            denom = pd.to_numeric(surface[surface_col], errors="coerce").replace(0, np.nan)
            rows[f"ratio_formula_surface_{col}"] = pd.to_numeric(base[col], errors="coerce") / denom
    for prop in ["en", "radius", "d", "valence", "Z"]:
        std_col = f"surface_{prop}_std"
        mean_col = f"surface_{prop}_mean"
        min_col = f"surface_{prop}_min"
        max_col = f"surface_{prop}_max"
        if min_col in surface and max_col in surface:
            rows[f"surface_{prop}_range"] = pd.to_numeric(surface[max_col], errors="coerce") - pd.to_numeric(surface[min_col], errors="coerce")
        if std_col in surface and mean_col in surface:
            denom = pd.to_numeric(surface[mean_col], errors="coerce").replace(0, np.nan)
            rows[f"surface_{prop}_relative_std"] = pd.to_numeric(surface[std_col], errors="coerce") / denom
    if "surface_d_std" in surface and "surface_en_std" in surface:
        rows["surface_electronic_mismatch"] = pd.to_numeric(surface["surface_d_std"], errors="coerce") + pd.to_numeric(surface["surface_en_std"], errors="coerce")
    return pd.DataFrame(rows, index=base.index)


def _parsed_site_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for site in df["site"]:
        text = str(site or "").lower()
        tokens = [token for token in re.split(r"[|/]+", text) if token]
        pattern = next((token.upper() for token in tokens if re.fullmatch(r"[AB](?:_[AB])+", token.upper())), "")
        letters = re.findall(r"[AB]", pattern)
        site_num = re.search(r"site\s*([0-9]+)", text)
        row = {
            "site_parse_is_fcc": float("fcc" in text),
            "site_parse_is_hcp": float("hcp" in text),
            "site_parse_is_tilt": float("tilt" in text),
            "site_parse_is_numbered": float(site_num is not None),
            "site_parse_number": float(site_num.group(1)) if site_num else 0.0,
            "site_pattern_A_count": float(letters.count("A")),
            "site_pattern_B_count": float(letters.count("B")),
            "site_pattern_length": float(len(letters)),
            "site_pattern_unique_count": float(len(set(letters))) if letters else 0.0,
            "site_pattern_is_AAA": float(pattern == "A_A_A"),
            "site_pattern_is_AAB": float(pattern == "A_A_B"),
            "site_pattern_is_ABB": float(pattern == "A_B_B"),
            "site_pattern_is_AA": float(pattern == "A_A"),
            "site_pattern_is_AB": float(pattern == "A_B"),
            "site_pattern_all_same": float(bool(letters) and len(set(letters)) == 1),
            "site_anchor_is_A": float(tokens[-1].upper() == "A") if tokens else 0.0,
            "site_anchor_is_B": float(tokens[-1].upper() == "B") if tokens else 0.0,
        }
        rows.append(row)
    return pd.DataFrame(rows, index=df.index)


def _categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    facet = df["facet"].fillna("unknown").astype(str).str.extract(r"(111|100|101|110|211|001|0001)", expand=False).fillna("other")
    site = df["site"].fillna("unknown").astype(str).str.lower()
    site_type = np.select(
        [
            site.str.contains("top|ontop", regex=True),
            site.str.contains("bridge|bri", regex=True),
            site.str.contains("hollow|fcc|hcp", regex=True),
            site.str.contains("4fold|four", regex=True),
        ],
        ["top", "bridge", "hollow", "fourfold"],
        default="other",
    )
    return pd.get_dummies(pd.DataFrame({"facet_cat": facet, "site_cat": site_type}), dtype=float)


def her_feature_frame(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    base = build_feature_frame(df, config)
    surface = _surface_features(df)
    cats = _categorical_features(df)
    interaction = _interaction_features(base, surface)
    parsed_site = _parsed_site_features(df)
    surface_labels = _surface_label_features(df)
    return pd.concat([base, surface, interaction, parsed_site, surface_labels, cats], axis=1)


def featurize_her_only(config: dict[str, Any], target_window: tuple[float, float], max_label_std: float) -> pd.DataFrame:
    raw = pd.read_csv(RAW_PATH)
    clean = strict_clean_her(raw, target_window=target_window, max_label_std=max_label_std)
    features = clean_features(her_feature_frame(clean, config))
    metadata = clean[["candidate_id", "formula", "surface", "facet", "site", "source", "fallback_used", "dG_H", "label_std", "label_count"]]
    out = pd.concat([metadata.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    FEATURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(FEATURE_PATH, index=False)
    print(f"Saved HER-only features: {out.shape} -> {FEATURE_PATH}")
    return out


def feature_columns(df: pd.DataFrame, top_n: int) -> list[str]:
    feats = [c for c in df.columns if c not in NON_FEATURE and pd.api.types.is_numeric_dtype(df[c])]
    if len(feats) <= top_n:
        return feats
    # Use variance ranking after median-imputed featurization. This is a cheap,
    # target-independent way to avoid very wide Magpie tables on CPU.
    return list(df[feats].var(numeric_only=True).sort_values(ascending=False).head(top_n).index)


def model_zoo(seed: int, n_jobs: int) -> dict[str, Any]:
    models: dict[str, Any] = {
        "Ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "Huber": Pipeline([("scaler", StandardScaler()), ("model", HuberRegressor(alpha=0.0001, max_iter=500))]),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=None, min_samples_leaf=2, random_state=seed, n_jobs=n_jobs
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=400, max_depth=None, min_samples_leaf=2, random_state=seed, n_jobs=n_jobs
        ),
    }
    if XGBRegressor is not None:
        models["XGBoost"] = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            n_jobs=n_jobs,
            tree_method="hist",
            verbosity=0,
        )
    return models


def evaluate_and_screen(df: pd.DataFrame, config: dict[str, Any], top_features: int) -> pd.DataFrame:
    seed = int(config["random_seed"])
    n_jobs = int(config["models"].get("n_jobs", 2))
    feats = feature_columns(df, top_features)
    x = df[feats]
    y = df["dG_H"]
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    rows = []
    fitted = {}
    scoring = {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"}
    for name, model in model_zoo(seed, n_jobs).items():
        scores = cross_validate(model, x, y, cv=cv, scoring=scoring, n_jobs=1)
        row = {
            "model": name,
            "mae_mean": -float(np.mean(scores["test_mae"])),
            "mae_std": float(np.std(-scores["test_mae"])),
            "rmse_mean": -float(np.mean(scores["test_rmse"])),
            "rmse_std": float(np.std(-scores["test_rmse"])),
            "r2_mean": float(np.mean(scores["test_r2"])),
            "r2_std": float(np.std(scores["test_r2"])),
            "n_samples": len(df),
            "n_features": len(feats),
        }
        rows.append(row)
        fitted[name] = model.fit(x, y)
        print(f"HER-only {name}: MAE={row['mae_mean']:.3f}, RMSE={row['rmse_mean']:.3f}, R2={row['r2_mean']:.3f}")

    metrics = pd.DataFrame(rows).sort_values("mae_mean")
    metrics["is_best"] = metrics["model"] == metrics.iloc[0]["model"]
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_PATH, index=False)

    best_name = str(metrics.iloc[0]["model"])
    best_model = fitted[best_name]
    pred = df[["candidate_id", "formula", "surface", "facet", "site", "source", "fallback_used", "dG_H", "label_std", "label_count"]].copy()
    pred["pred_dG_H"] = best_model.predict(x)
    PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(PRED_PATH, index=False)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
    plot_model = model_zoo(seed, n_jobs)[best_name].fit(x_train, y_train)
    y_pred = plot_model.predict(x_test)
    _save_parity(y_test.to_numpy(), y_pred, best_name)
    _save_volcano(pred)

    ranked = pred.copy()
    ranked["screen_score"] = ranked["pred_dG_H"].abs()
    ranked = ranked.sort_values("screen_score").head(50)
    RANKING_PATH.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(RANKING_PATH, index=False)
    print(f"Saved metrics -> {METRICS_PATH}")
    print(f"Saved top HER-only ranking -> {RANKING_PATH}")
    print(f"Best HER-only model: {best_name}")
    print(f"Top candidate: {ranked.iloc[0]['candidate_id']} pred_dG_H={ranked.iloc[0]['pred_dG_H']:.4f}")
    return metrics


def _save_parity(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> None:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    lo = min(float(y_true.min()), float(y_pred.min()))
    hi = max(float(y_true.max()), float(y_pred.max()))
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.scatter(y_true, y_pred, s=12, alpha=0.55, color="#2f6f8f")
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    ax.set_xlabel("True normalized Delta G_H* (eV)")
    ax.set_ylabel("Predicted Delta G_H* (eV)")
    ax.set_title(f"HER-only parity: {model_name}")
    ax.text(0.05, 0.95, f"MAE={mae:.3f}\nRMSE={rmse:.3f}\nR2={r2:.3f}", transform=ax.transAxes, va="top")
    fig.tight_layout()
    PARITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PARITY_PATH, dpi=300)
    plt.close(fig)


def _save_volcano(pred: pd.DataFrame) -> None:
    df = pred.copy()
    df["activity_proxy"] = -df["pred_dG_H"].abs()
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.scatter(df["pred_dG_H"], df["activity_proxy"], s=14, alpha=0.55, color="#31688e")
    ax.axvline(0.0, linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("Predicted Delta G_H* (eV)")
    ax.set_ylabel("Activity proxy: -|Delta G_H*|")
    ax.set_title("HER-only volcano screening")
    fig.tight_layout()
    VOLCANO_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(VOLCANO_PATH, dpi=300)
    plt.close(fig)


def run(config_path: str | Path, target_min: float, target_max: float, max_label_std: float, top_features: int) -> pd.DataFrame:
    config = load_config(config_path)
    df = featurize_her_only(config, target_window=(target_min, target_max), max_label_std=max_label_std)
    return evaluate_and_screen(df, config, top_features=top_features)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run improved HER-only modeling.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--target-min", type=float, default=-2.5)
    parser.add_argument("--target-max", type=float, default=2.5)
    parser.add_argument("--max-label-std", type=float, default=0.4)
    parser.add_argument("--top-features", type=int, default=280)
    args = parser.parse_args()
    run(args.config, args.target_min, args.target_max, args.max_label_std, args.top_features)


if __name__ == "__main__":
    main()
