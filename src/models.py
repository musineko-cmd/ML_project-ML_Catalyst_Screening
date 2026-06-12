"""Train and evaluate tabular regressors for HER and OER targets."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - dependency installed by requirements.
    XGBRegressor = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.yaml"
NON_FEATURE = {
    "source_id",
    "candidate_id",
    "formula",
    "surface",
    "facet",
    "site",
    "source",
    "fallback_used",
    "dG_H",
    "dG_OH",
    "dG_O",
    "dG_OOH",
    "overpotential",
    "dG_O_minus_OH",
}
TARGETS = {"her": ["dG_H"], "oer": ["dG_OH", "dG_O", "dG_OOH", "overpotential"]}


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE and pd.api.types.is_numeric_dtype(df[c])]


def select_feature_columns(df: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    feats = feature_columns(df)
    top_n = int(config["models"].get("feature_selection_top_n", len(feats)))
    if len(feats) <= top_n:
        return feats
    variances = df[feats].var(numeric_only=True).sort_values(ascending=False)
    return list(variances.head(top_n).index)


def _grid(config_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(config_grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(config_grid[k] for k in keys))]


def build_models(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config["random_seed"])
    n_jobs = int(config["models"]["n_jobs"])
    mlp_cfg = config["models"]["mlp"]
    models: dict[str, Any] = {
        "LinearRegression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "Ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "RandomForest": RandomForestRegressor(random_state=seed, n_jobs=n_jobs),
        "MLP": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=tuple(mlp_cfg["hidden_layer_sizes"]),
                        alpha=float(mlp_cfg["alpha"]),
                        max_iter=int(mlp_cfg["max_iter"]),
                        random_state=seed,
                        early_stopping=True,
                    ),
                ),
            ]
        ),
    }
    if XGBRegressor is not None:
        models["XGBoost"] = XGBRegressor(
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=n_jobs,
            tree_method="hist",
            verbosity=0,
        )
    return models


def tune_if_needed(name: str, base_model: Any, x: pd.DataFrame, y: pd.Series, config: dict[str, Any], cv: KFold) -> Any:
    if name == "RandomForest":
        candidates = _grid(config["models"]["rf_grid"])
    elif name == "XGBoost":
        candidates = _grid(config["models"]["xgb_grid"])
    else:
        return clone(base_model)

    best_score = np.inf
    best_model = None
    for params in candidates:
        model = clone(base_model).set_params(**params)
        scores = cross_validate(model, x, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=1)
        mae = -float(np.mean(scores["test_score"]))
        if mae < best_score:
            best_score = mae
            best_model = model
    assert best_model is not None
    return best_model


def cv_metrics(model: Any, x: pd.DataFrame, y: pd.Series, cv: KFold) -> dict[str, float]:
    scoring = {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"}
    scores = cross_validate(model, x, y, cv=cv, scoring=scoring, n_jobs=1)
    return {
        "mae_mean": -float(np.mean(scores["test_mae"])),
        "mae_std": float(np.std(-scores["test_mae"])),
        "rmse_mean": -float(np.mean(scores["test_rmse"])),
        "rmse_std": float(np.std(-scores["test_rmse"])),
        "r2_mean": float(np.mean(scores["test_r2"])),
        "r2_std": float(np.std(scores["test_r2"])),
    }


def parity_plot(model: Any, x: pd.DataFrame, y: pd.Series, reaction: str, target: str, model_name: str, config: dict[str, Any]) -> None:
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=float(config["models"]["test_size_for_plots"]),
        random_state=int(config["random_seed"]),
    )
    fitted = clone(model).fit(x_train, y_train)
    pred = fitted.predict(x_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = root_mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    lo = min(float(np.min(y_test)), float(np.min(pred)))
    hi = max(float(np.max(y_test)), float(np.max(pred)))
    fig, ax = plt.subplots(figsize=(4.4, 4.1))
    ax.scatter(y_test, pred, s=24, alpha=0.75, color="#315f91")
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    ax.set_xlabel(f"True {target}")
    ax.set_ylabel(f"Predicted {target}")
    ax.set_title(f"{reaction.upper()} {target}: {model_name}")
    ax.text(0.05, 0.95, f"MAE={mae:.3f}\nRMSE={rmse:.3f}\nR2={r2:.3f}", transform=ax.transAxes, va="top")
    fig.tight_layout()
    out = Path(config["paths"]["figures_dir"]) / f"parity_{reaction}_{target}.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)


def save_importance(model: Any, features: list[str], reaction: str, target: str, model_name: str, config: dict[str, Any]) -> None:
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    else:
        return
    out = pd.DataFrame({"feature": features, "importance": values}).sort_values("importance", ascending=False)
    path = Path(config["paths"]["metrics_dir"]) / f"feature_importance_{reaction}_{target}_{model_name}.csv"
    out.to_csv(path, index=False)


def fit_target(
    df: pd.DataFrame,
    reaction: str,
    target: str,
    config: dict[str, Any],
    predictions: pd.DataFrame,
) -> list[dict[str, Any]]:
    feats = select_feature_columns(df, config)
    data = df.dropna(subset=[target]).copy()
    max_n = int(config["models"]["max_samples_per_target"])
    if len(data) > max_n:
        data = data.sample(max_n, random_state=int(config["random_seed"]))
    x = data[feats]
    y = data[target]
    cv = KFold(n_splits=int(config["models"]["cv_folds"]), shuffle=True, random_state=int(config["random_seed"]))
    rows: list[dict[str, Any]] = []
    fitted_models: dict[str, Any] = {}

    for name, base in build_models(config).items():
        tuned = tune_if_needed(name, base, x, y, config, cv)
        metrics = cv_metrics(tuned, x, y, cv)
        fitted = clone(tuned).fit(x, y)
        fitted_models[name] = fitted
        if name in {"RandomForest", "XGBoost"}:
            save_importance(fitted, feats, reaction, target, name, config)
        rows.append({"reaction": reaction, "target": target, "model": name, **metrics})
        print(f"{reaction.upper()} {target} {name}: MAE={metrics['mae_mean']:.3f}, R2={metrics['r2_mean']:.3f}")

    best = min(rows, key=lambda r: r["mae_mean"])
    best_model = fitted_models[best["model"]]
    parity_plot(best_model, x, y, reaction, target, best["model"], config)
    predictions[f"pred_{target}"] = best_model.predict(df[feats])
    for row in rows:
        row["is_best"] = row["model"] == best["model"]
    return rows


def train_models(config_path: str | Path = DEFAULT_CONFIG) -> pd.DataFrame:
    config = load_config(config_path)
    Path(config["paths"]["metrics_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["paths"]["figures_dir"]).mkdir(parents=True, exist_ok=True)
    processed = Path(config["paths"]["processed_dir"])
    all_rows: list[dict[str, Any]] = []

    for reaction in ["her", "oer"]:
        df = pd.read_csv(processed / f"{reaction}_features.csv")
        keep_cols = [c for c in ["candidate_id", "formula", "surface", "facet", "source", "fallback_used"] if c in df]
        predictions = df[keep_cols + TARGETS[reaction]].copy()
        for target in TARGETS[reaction]:
            all_rows.extend(fit_target(df, reaction, target, config, predictions))
        pred_path = Path(config["paths"]["metrics_dir"]) / f"predictions_{reaction}.csv"
        predictions.to_csv(pred_path, index=False)
        print(f"Saved predictions to {pred_path}")

    metrics = pd.DataFrame(all_rows)
    metrics_path = Path(config["paths"]["metrics_dir"]) / "cv_scores.csv"
    metrics.to_csv(metrics_path, index=False)
    print(f"Saved CV metrics to {metrics_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate HER/OER regressors.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    train_models(args.config)


if __name__ == "__main__":
    main()
