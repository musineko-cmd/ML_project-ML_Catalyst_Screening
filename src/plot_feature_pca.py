"""Visualize HER feature space with PCA.

The PCA is used only for interpretation/plots, not for model training. It fits a
2D PCA on the same selected HER features used for generated-candidate prediction,
then projects the training set, all generated candidates, top-10 lists, and any
DFT-converged candidates into the same space.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from featurize import load_config
from generate_candidates import featurize_candidates
from her_only import FEATURE_PATH, feature_columns
from predict_generated_candidates import normalize_generated_candidates


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.yaml"
DEFAULT_GENERATED = ROOT / "generated_candidates.csv"
DEFAULT_PREDICTIONS = ROOT / "results/metrics/generated_candidates_extratrees_predictions_updated_features.csv"
DEFAULT_NEAR_ZERO_TOP = ROOT / "results/rankings/generated_candidates_top10_extratrees_near_zero_updated_features.csv"
DEFAULT_PRIORITY_TOP = ROOT / "results/rankings/generated_candidates_top10_extratrees_priority_updated_features.csv"
DEFAULT_FIGURE = ROOT / "results/figures/her_feature_pca_generated_candidates.png"
DEFAULT_COORDS = ROOT / "results/metrics/her_feature_pca_coordinates.csv"


def _read_optional_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _dft_converged_candidates() -> set[str]:
    candidates: set[str] = set()
    for path in [
        ROOT / "dft_runs_mp/dft_h_adsorption_converged_only.csv",
        ROOT / "dft_runs_uncertainty_top10/dft_h_adsorption_converged_only.csv",
    ]:
        df = _read_optional_csv(path)
        if "candidate_id" in df:
            candidates.update(df["candidate_id"].dropna().astype(str))
    return candidates


def _rank_map(df: pd.DataFrame) -> dict[str, int]:
    if df.empty or "candidate_id" not in df:
        return {}
    return {candidate_id: idx + 1 for idx, candidate_id in enumerate(df["candidate_id"].astype(str))}


def build_pca_frame(
    config_path: Path,
    generated_path: Path,
    predictions_path: Path,
    near_zero_top_path: Path,
    priority_top_path: Path,
    top_features: int,
    train_sample: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    config = load_config(config_path)
    train = pd.read_csv(FEATURE_PATH)
    raw_generated = pd.read_csv(generated_path)
    predictions = pd.read_csv(predictions_path)
    near_zero_top = _read_optional_csv(near_zero_top_path)
    priority_top = _read_optional_csv(priority_top_path)

    feats = feature_columns(train, top_features)
    generated_meta = normalize_generated_candidates(raw_generated)
    x_train = train[feats]
    x_generated = featurize_candidates(generated_meta, config, train, feats)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_generated_scaled = scaler.transform(x_generated)
    pca = PCA(n_components=2, random_state=int(config.get("random_seed", 42)))
    train_pca = pca.fit_transform(x_train_scaled)
    generated_pca = pca.transform(x_generated_scaled)

    rng = np.random.default_rng(int(config.get("random_seed", 42)))
    if len(train) > train_sample:
        train_idx = np.sort(rng.choice(len(train), train_sample, replace=False))
    else:
        train_idx = np.arange(len(train))

    train_frame = pd.DataFrame(
        {
            "kind": "train",
            "candidate_id": train.iloc[train_idx]["candidate_id"].astype(str).to_numpy(),
            "formula": train.iloc[train_idx]["formula"].astype(str).to_numpy(),
            "surface": train.iloc[train_idx]["surface"].astype(str).to_numpy(),
            "site": train.iloc[train_idx]["site"].astype(str).to_numpy(),
            "PC1": train_pca[train_idx, 0],
            "PC2": train_pca[train_idx, 1],
            "dG_H": train.iloc[train_idx]["dG_H"].to_numpy(),
            "pred_dG_H": np.nan,
            "pred_sigma": np.nan,
            "near_zero_rank": np.nan,
            "priority_rank": np.nan,
            "dft_converged": False,
        }
    )

    pred_by_id = predictions.set_index("candidate_id") if "candidate_id" in predictions else pd.DataFrame()
    near_rank = _rank_map(near_zero_top)
    priority_rank = _rank_map(priority_top)
    dft_converged = _dft_converged_candidates()

    generated_frame = pd.DataFrame(
        {
            "kind": "generated",
            "candidate_id": generated_meta["candidate_id"].astype(str),
            "formula": generated_meta["formula"].astype(str),
            "surface": generated_meta["surface"].astype(str),
            "site": generated_meta["site"].astype(str),
            "PC1": generated_pca[:, 0],
            "PC2": generated_pca[:, 1],
            "dG_H": np.nan,
        }
    )
    if not pred_by_id.empty:
        generated_frame["pred_dG_H"] = generated_frame["candidate_id"].map(pred_by_id["pred_dG_H_ExtraTrees"])
        generated_frame["pred_sigma"] = generated_frame["candidate_id"].map(pred_by_id["pred_dG_H_sigma_ExtraTrees"])
    else:
        generated_frame["pred_dG_H"] = np.nan
        generated_frame["pred_sigma"] = np.nan
    generated_frame["near_zero_rank"] = generated_frame["candidate_id"].map(near_rank)
    generated_frame["priority_rank"] = generated_frame["candidate_id"].map(priority_rank)
    generated_frame["dft_converged"] = generated_frame["candidate_id"].isin(dft_converged)

    return pd.concat([train_frame, generated_frame], ignore_index=True), pca.explained_variance_ratio_


def plot_pca(frame: pd.DataFrame, explained: np.ndarray, figure_path: Path) -> None:
    train = frame[frame["kind"] == "train"]
    generated = frame[frame["kind"] == "generated"]
    top = generated[generated["near_zero_rank"].notna() | generated["priority_rank"].notna()].copy()
    dft = generated[generated["dft_converged"]].copy()

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    train_scatter = ax.scatter(
        train["PC1"],
        train["PC2"],
        c=train["dG_H"],
        cmap="coolwarm",
        vmin=-1.5,
        vmax=1.5,
        s=9,
        alpha=0.28,
        linewidths=0,
        label="HER training data",
    )
    ax.scatter(
        generated["PC1"],
        generated["PC2"],
        c="black",
        s=20,
        alpha=0.35,
        marker="o",
        label="generated candidates",
    )
    if not top.empty:
        colors = top["pred_dG_H"].fillna(0.0)
        ax.scatter(
            top["PC1"],
            top["PC2"],
            c=colors,
            cmap="PiYG_r",
            vmin=-0.2,
            vmax=0.2,
            s=110,
            marker="*",
            edgecolors="black",
            linewidths=0.8,
            label="near-zero or priority top 10",
            zorder=4,
        )
    if not dft.empty:
        ax.scatter(
            dft["PC1"],
            dft["PC2"],
            facecolors="none",
            edgecolors="#ffb000",
            s=210,
            marker="o",
            linewidths=2.0,
            label="DFT converged candidates",
            zorder=5,
        )

    for _, row in top.iterrows():
        labels = []
        if pd.notna(row["near_zero_rank"]):
            labels.append(f"N{int(row['near_zero_rank'])}")
        if pd.notna(row["priority_rank"]):
            labels.append(f"P{int(row['priority_rank'])}")
        ax.annotate(
            "/".join(labels),
            (row["PC1"], row["PC2"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
            color="black",
        )

    cbar = fig.colorbar(train_scatter, ax=ax, pad=0.02)
    cbar.set_label("training true dG_H (eV)")
    ax.axhline(0, color="0.85", lw=0.8)
    ax.axvline(0, color="0.85", lw=0.8)
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% variance)")
    ax.set_title("HER Feature-Space PCA: Training Data vs Generated Candidates")
    ax.legend(loc="best", frameon=True, fontsize=8)
    ax.grid(alpha=0.15)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot PCA of HER feature space and generated candidates.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--generated", type=Path, default=DEFAULT_GENERATED)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--near-zero-top", type=Path, default=DEFAULT_NEAR_ZERO_TOP)
    parser.add_argument("--priority-top", type=Path, default=DEFAULT_PRIORITY_TOP)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--coords", type=Path, default=DEFAULT_COORDS)
    parser.add_argument("--top-features", type=int, default=280)
    parser.add_argument("--train-sample", type=int, default=5000)
    args = parser.parse_args()

    frame, explained = build_pca_frame(
        config_path=args.config,
        generated_path=args.generated,
        predictions_path=args.predictions,
        near_zero_top_path=args.near_zero_top,
        priority_top_path=args.priority_top,
        top_features=args.top_features,
        train_sample=args.train_sample,
    )
    args.coords.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.coords, index=False)
    plot_pca(frame, explained, args.figure)
    print(f"Saved PCA coordinates -> {args.coords}")
    print(f"Saved PCA figure -> {args.figure}")
    print(f"Explained variance: PC1={explained[0]:.4f}, PC2={explained[1]:.4f}")


if __name__ == "__main__":
    main()
