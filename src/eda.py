"""Exploratory plots for HER/OER feature matrices."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.yaml"
TARGETS = {"her": ["dG_H"], "oer": ["dG_OH", "dG_O", "dG_OOH", "overpotential"]}
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


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE and pd.api.types.is_numeric_dtype(df[c])]


def save_histograms(df: pd.DataFrame, targets: list[str], name: str, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(targets), figsize=(4.2 * len(targets), 3.2))
    if len(targets) == 1:
        axes = [axes]
    for ax, target in zip(axes, targets):
        ax.hist(df[target].dropna(), bins=30, color="#3b6ea8", edgecolor="white")
        ax.set_title(f"{name.upper()} {target}")
        ax.set_xlabel("Energy / V")
        ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_target_histograms.png", dpi=300)
    plt.close(fig)


def save_correlation_heatmap(df: pd.DataFrame, target: str, name: str, out_dir: Path, top_n: int) -> None:
    feats = feature_columns(df)
    corr = df[feats + [target]].corr(numeric_only=True)[target].drop(labels=[target]).abs().sort_values(ascending=False)
    selected = list(corr.head(top_n).index) + [target]
    matrix = df[selected].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(matrix.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(selected)))
    ax.set_yticks(range(len(selected)))
    ax.set_xticklabels(selected, rotation=60, ha="right", fontsize=7)
    ax.set_yticklabels(selected, fontsize=7)
    ax.set_title(f"{name.upper()} top feature correlations with {target}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_{target}_correlation_heatmap.png", dpi=300)
    plt.close(fig)


def _pca_2d(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = x - x.mean(axis=0, keepdims=True)
    scale = x.std(axis=0, keepdims=True)
    scale[scale == 0] = 1.0
    x = x / scale
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    coords = x @ vt[:2].T
    denom = max(float((s**2).sum()), 1e-12)
    explained = (s[:2] ** 2) / denom
    return coords, explained


def save_pca(df: pd.DataFrame, target: str, name: str, out_dir: Path, max_features: int) -> None:
    feats = feature_columns(df)[:max_features]
    x = df[feats].to_numpy(dtype=float)
    coords, explained = _pca_2d(x)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=df[target], cmap="viridis", s=18, alpha=0.8)
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
    ax.set_title(f"{name.upper()} feature PCA colored by {target}")
    fig.colorbar(sc, ax=ax, label=target)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_{target}_pca.png", dpi=300)
    plt.close(fig)


def run_eda(config_path: str | Path = DEFAULT_CONFIG) -> None:
    config = load_config(config_path)
    out_dir = Path(config["paths"]["figures_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    processed = Path(config["paths"]["processed_dir"])
    her = pd.read_csv(processed / "her_features.csv")
    oer = pd.read_csv(processed / "oer_features.csv")
    top_n = int(config["features"]["correlation_plot_top_n"])
    max_features = int(config["features"]["pca_max_features"])

    for name, df in [("her", her), ("oer", oer)]:
        save_histograms(df, TARGETS[name], name, out_dir)
        for target in TARGETS[name]:
            save_correlation_heatmap(df, target, name, out_dir, top_n)
        save_pca(df, TARGETS[name][-1], name, out_dir, max_features)
    print(f"Saved EDA figures to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate EDA figures.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    run_eda(args.config)


if __name__ == "__main__":
    main()
