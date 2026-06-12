"""Volcano-style screening and ranking using model predictions."""

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


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def her_screen(pred: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    out_dir = Path(config["paths"]["figures_dir"])
    rank_dir = Path(config["paths"]["rankings_dir"])
    group_cols = [c for c in ["candidate_id", "formula", "surface", "facet", "source", "fallback_used"] if c in pred.columns]
    numeric_cols = [c for c in ["dG_H", "pred_dG_H"] if c in pred.columns]
    df = pred.groupby(group_cols, as_index=False)[numeric_cols].mean()
    df["her_activity_proxy"] = -df["pred_dG_H"].abs()
    df["screen_score"] = df["pred_dG_H"].abs()
    ranked = df.sort_values("screen_score", ascending=True)
    top = ranked.head(int(config["screening"]["top_n"]))
    top.to_csv(rank_dir / "her_top.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.scatter(df["pred_dG_H"], df["her_activity_proxy"], s=22, alpha=0.75, color="#2b6c7f")
    ax.axvline(0.0, color="black", linewidth=1, linestyle="--", label="Volcano optimum")
    ax.set_xlabel("Predicted Delta G_H* (eV)")
    ax.set_ylabel("Activity proxy: -|Delta G_H*|")
    ax.set_title("HER volcano screening")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "her_volcano.png", dpi=300)
    plt.close(fig)
    return top


def oer_screen(pred: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    out_dir = Path(config["paths"]["figures_dir"])
    rank_dir = Path(config["paths"]["rankings_dir"])
    group_cols = [c for c in ["candidate_id", "formula", "surface", "facet", "source", "fallback_used"] if c in pred.columns]
    numeric_cols = [
        c
        for c in [
            "dG_OH",
            "dG_O",
            "dG_OOH",
            "overpotential",
            "pred_dG_OH",
            "pred_dG_O",
            "pred_dG_OOH",
            "pred_overpotential",
        ]
        if c in pred.columns
    ]
    df = pred.groupby(group_cols, as_index=False)[numeric_cols].mean()
    df["pred_dG_O_minus_OH"] = df["pred_dG_O"] - df["pred_dG_OH"]
    df["screen_score"] = df["pred_overpotential"]
    ranked = df.sort_values("screen_score", ascending=True)
    top = ranked.head(int(config["screening"]["top_n"]))
    top.to_csv(rank_dir / "oer_top.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    sc = ax.scatter(
        df["pred_dG_O_minus_OH"],
        df["pred_overpotential"],
        c=df["pred_overpotential"],
        cmap="magma_r",
        s=24,
        alpha=0.8,
    )
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Predicted Delta G_O* - Delta G_OH* (eV)")
    ax.set_ylabel("Predicted OER overpotential (V)")
    ax.set_title("OER descriptor volcano screening")
    fig.colorbar(sc, ax=ax, label="eta (V)")
    fig.tight_layout()
    fig.savefig(out_dir / "oer_volcano.png", dpi=300)
    plt.close(fig)
    return top


def run_screening(config_path: str | Path = DEFAULT_CONFIG) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = load_config(config_path)
    for key in ["figures_dir", "rankings_dir"]:
        Path(config["paths"][key]).mkdir(parents=True, exist_ok=True)
    metrics_dir = Path(config["paths"]["metrics_dir"])
    her_pred = pd.read_csv(metrics_dir / "predictions_her.csv")
    oer_pred = pd.read_csv(metrics_dir / "predictions_oer.csv")
    her_top = her_screen(her_pred, config)
    oer_top = oer_screen(oer_pred, config)
    print(f"Saved HER/OER rankings to {config['paths']['rankings_dir']}")
    print(f"Top HER candidate: {her_top.iloc[0]['candidate_id']}")
    print(f"Top OER candidate: {oer_top.iloc[0]['candidate_id']}")
    return her_top, oer_top


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank HER/OER candidates and draw volcano plots.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    run_screening(args.config)


if __name__ == "__main__":
    main()
