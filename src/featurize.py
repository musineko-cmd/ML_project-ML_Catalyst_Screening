"""Featurize HER/OER reaction tables into tabular ML matrices."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.yaml"

# Compact periodic table subset covering common catalyst elements. Values are
# approximate and sufficient for transparent hand-crafted descriptors.
ELEMENT_PROPS: dict[str, dict[str, float]] = {
    "H": {"Z": 1, "en": 2.20, "radius": 53, "group": 1, "period": 1, "valence": 1, "d": 0, "ox": 1},
    "C": {"Z": 6, "en": 2.55, "radius": 67, "group": 14, "period": 2, "valence": 4, "d": 0, "ox": 4},
    "N": {"Z": 7, "en": 3.04, "radius": 56, "group": 15, "period": 2, "valence": 5, "d": 0, "ox": -3},
    "O": {"Z": 8, "en": 3.44, "radius": 48, "group": 16, "period": 2, "valence": 6, "d": 0, "ox": -2},
    "Na": {"Z": 11, "en": 0.93, "radius": 190, "group": 1, "period": 3, "valence": 1, "d": 0, "ox": 1},
    "Mg": {"Z": 12, "en": 1.31, "radius": 145, "group": 2, "period": 3, "valence": 2, "d": 0, "ox": 2},
    "Al": {"Z": 13, "en": 1.61, "radius": 118, "group": 13, "period": 3, "valence": 3, "d": 0, "ox": 3},
    "Si": {"Z": 14, "en": 1.90, "radius": 111, "group": 14, "period": 3, "valence": 4, "d": 0, "ox": 4},
    "P": {"Z": 15, "en": 2.19, "radius": 98, "group": 15, "period": 3, "valence": 5, "d": 0, "ox": 5},
    "S": {"Z": 16, "en": 2.58, "radius": 88, "group": 16, "period": 3, "valence": 6, "d": 0, "ox": -2},
    "K": {"Z": 19, "en": 0.82, "radius": 243, "group": 1, "period": 4, "valence": 1, "d": 0, "ox": 1},
    "Ca": {"Z": 20, "en": 1.00, "radius": 194, "group": 2, "period": 4, "valence": 2, "d": 0, "ox": 2},
    "Sc": {"Z": 21, "en": 1.36, "radius": 184, "group": 3, "period": 4, "valence": 3, "d": 1, "ox": 3},
    "Ti": {"Z": 22, "en": 1.54, "radius": 176, "group": 4, "period": 4, "valence": 4, "d": 2, "ox": 4},
    "V": {"Z": 23, "en": 1.63, "radius": 171, "group": 5, "period": 4, "valence": 5, "d": 3, "ox": 5},
    "Cr": {"Z": 24, "en": 1.66, "radius": 166, "group": 6, "period": 4, "valence": 6, "d": 5, "ox": 3},
    "Mn": {"Z": 25, "en": 1.55, "radius": 161, "group": 7, "period": 4, "valence": 7, "d": 5, "ox": 4},
    "Fe": {"Z": 26, "en": 1.83, "radius": 156, "group": 8, "period": 4, "valence": 8, "d": 6, "ox": 3},
    "Co": {"Z": 27, "en": 1.88, "radius": 152, "group": 9, "period": 4, "valence": 9, "d": 7, "ox": 3},
    "Ni": {"Z": 28, "en": 1.91, "radius": 149, "group": 10, "period": 4, "valence": 10, "d": 8, "ox": 2},
    "Cu": {"Z": 29, "en": 1.90, "radius": 145, "group": 11, "period": 4, "valence": 11, "d": 10, "ox": 2},
    "Zn": {"Z": 30, "en": 1.65, "radius": 142, "group": 12, "period": 4, "valence": 12, "d": 10, "ox": 2},
    "Ga": {"Z": 31, "en": 1.81, "radius": 136, "group": 13, "period": 4, "valence": 3, "d": 10, "ox": 3},
    "Ge": {"Z": 32, "en": 2.01, "radius": 125, "group": 14, "period": 4, "valence": 4, "d": 10, "ox": 4},
    "Sr": {"Z": 38, "en": 0.95, "radius": 219, "group": 2, "period": 5, "valence": 2, "d": 0, "ox": 2},
    "Y": {"Z": 39, "en": 1.22, "radius": 212, "group": 3, "period": 5, "valence": 3, "d": 1, "ox": 3},
    "Zr": {"Z": 40, "en": 1.33, "radius": 206, "group": 4, "period": 5, "valence": 4, "d": 2, "ox": 4},
    "Nb": {"Z": 41, "en": 1.60, "radius": 198, "group": 5, "period": 5, "valence": 5, "d": 4, "ox": 5},
    "Mo": {"Z": 42, "en": 2.16, "radius": 190, "group": 6, "period": 5, "valence": 6, "d": 5, "ox": 6},
    "Tc": {"Z": 43, "en": 1.90, "radius": 183, "group": 7, "period": 5, "valence": 7, "d": 5, "ox": 4},
    "Ru": {"Z": 44, "en": 2.20, "radius": 178, "group": 8, "period": 5, "valence": 8, "d": 7, "ox": 4},
    "Rh": {"Z": 45, "en": 2.28, "radius": 173, "group": 9, "period": 5, "valence": 9, "d": 8, "ox": 3},
    "Pd": {"Z": 46, "en": 2.20, "radius": 169, "group": 10, "period": 5, "valence": 10, "d": 10, "ox": 2},
    "Ag": {"Z": 47, "en": 1.93, "radius": 165, "group": 11, "period": 5, "valence": 11, "d": 10, "ox": 1},
    "Cd": {"Z": 48, "en": 1.69, "radius": 161, "group": 12, "period": 5, "valence": 12, "d": 10, "ox": 2},
    "In": {"Z": 49, "en": 1.78, "radius": 156, "group": 13, "period": 5, "valence": 3, "d": 10, "ox": 3},
    "Sn": {"Z": 50, "en": 1.96, "radius": 145, "group": 14, "period": 5, "valence": 4, "d": 10, "ox": 4},
    "Ba": {"Z": 56, "en": 0.89, "radius": 253, "group": 2, "period": 6, "valence": 2, "d": 0, "ox": 2},
    "La": {"Z": 57, "en": 1.10, "radius": 195, "group": 3, "period": 6, "valence": 3, "d": 1, "ox": 3},
    "Hf": {"Z": 72, "en": 1.30, "radius": 208, "group": 4, "period": 6, "valence": 4, "d": 2, "ox": 4},
    "Ta": {"Z": 73, "en": 1.50, "radius": 200, "group": 5, "period": 6, "valence": 5, "d": 3, "ox": 5},
    "W": {"Z": 74, "en": 2.36, "radius": 193, "group": 6, "period": 6, "valence": 6, "d": 4, "ox": 6},
    "Re": {"Z": 75, "en": 1.90, "radius": 188, "group": 7, "period": 6, "valence": 7, "d": 5, "ox": 4},
    "Os": {"Z": 76, "en": 2.20, "radius": 185, "group": 8, "period": 6, "valence": 8, "d": 6, "ox": 4},
    "Ir": {"Z": 77, "en": 2.20, "radius": 180, "group": 9, "period": 6, "valence": 9, "d": 7, "ox": 4},
    "Pt": {"Z": 78, "en": 2.28, "radius": 177, "group": 10, "period": 6, "valence": 10, "d": 9, "ox": 2},
    "Au": {"Z": 79, "en": 2.54, "radius": 174, "group": 11, "period": 6, "valence": 11, "d": 10, "ox": 3},
    "Hg": {"Z": 80, "en": 2.00, "radius": 171, "group": 12, "period": 6, "valence": 12, "d": 10, "ox": 2},
    "Pb": {"Z": 82, "en": 2.33, "radius": 154, "group": 14, "period": 6, "valence": 4, "d": 10, "ox": 2},
}

METADATA_COLUMNS = ["source_id", "candidate_id", "formula", "surface", "facet", "site", "source", "fallback_used"]


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_formula(formula: Any) -> dict[str, float]:
    """Parse a simple chemical formula into element amounts.

    The fallback parser intentionally handles the formula strings used here
    (e.g., ``Fe2NiO4``) and skips decorations after dashes from surface labels.
    """

    text = str(formula or "").split("-")[0]
    parts = re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)", text)
    comp: dict[str, float] = {}
    for element, amount in parts:
        comp[element] = comp.get(element, 0.0) + float(amount or 1.0)
    return comp


def _weighted_stats(values: np.ndarray, weights: np.ndarray) -> tuple[float, float, float, float]:
    if values.size == 0 or weights.sum() <= 0:
        return np.nan, np.nan, np.nan, np.nan
    mean = float(np.average(values, weights=weights))
    var = float(np.average((values - mean) ** 2, weights=weights))
    return mean, float(np.sqrt(var)), float(np.min(values)), float(np.max(values))


def composition_features(formula: Any) -> dict[str, float]:
    comp = parse_formula(formula)
    known = [(el, amt, ELEMENT_PROPS[el]) for el, amt in comp.items() if el in ELEMENT_PROPS and amt > 0]
    feats: dict[str, float] = {
        "n_elements": float(len(comp)),
        "n_known_elements": float(len(known)),
        "total_atoms": float(sum(comp.values()) or np.nan),
        "oxygen_fraction": float(comp.get("O", 0.0) / sum(comp.values())) if sum(comp.values()) else np.nan,
        "contains_oxygen": float("O" in comp),
    }
    if not known:
        return feats
    weights = np.array([amt for _, amt, _ in known], dtype=float)
    for prop in ["Z", "en", "radius", "group", "period", "valence", "d", "ox"]:
        values = np.array([props[prop] for _, _, props in known], dtype=float)
        mean, std, vmin, vmax = _weighted_stats(values, weights)
        feats[f"{prop}_mean"] = mean
        feats[f"{prop}_std"] = std
        feats[f"{prop}_min"] = vmin
        feats[f"{prop}_max"] = vmax
    # Simple d-band proxy: late/high-electronegativity transition metals tend to
    # bind adsorbates differently; this proxy is for interpretability only.
    feats["d_band_proxy"] = feats.get("d_mean", np.nan) - 0.35 * (feats.get("en_mean", np.nan) - 2.0)
    feats["metal_atom_fraction"] = float(sum(amt for el, amt in comp.items() if el != "O") / sum(comp.values()))
    return feats


def site_features(site: Any, facet: Any) -> dict[str, float]:
    text = str(site or "").lower()
    facet_text = str(facet or "")
    top = float("top" in text)
    bridge = float("bridge" in text or "bri" in text)
    hollow = float("hollow" in text or "fcc" in text or "hcp" in text)
    if top:
        coordination = 1.0
    elif bridge:
        coordination = 2.0
    elif hollow:
        coordination = 3.0
    else:
        coordination = 0.0
    digits = [int(ch) for ch in facet_text if ch.isdigit()]
    return {
        "site_is_top": top,
        "site_is_bridge": bridge,
        "site_is_hollow": hollow,
        "coordination_number": coordination,
        "facet_digit_sum": float(sum(digits)) if digits else 0.0,
    }


def _try_matminer_features(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if not config.get("features", {}).get("use_matminer_if_available", True):
        return pd.DataFrame(index=df.index)
    try:  # pragma: no cover - optional heavy dependencies are environment-specific.
        from matminer.featurizers.composition import ElementProperty
        from pymatgen.core import Composition

        temp = pd.DataFrame({"composition": [Composition(f) for f in df["formula"]]}, index=df.index)
        featurizer = ElementProperty.from_preset("magpie")
        temp = featurizer.featurize_dataframe(temp, "composition", ignore_errors=True)
        temp = temp.drop(columns=["composition"])
        temp = temp.add_prefix("matminer_")
        return temp.apply(pd.to_numeric, errors="coerce")
    except Exception as exc:
        print(f"WARNING: matminer/pymatgen featurization unavailable; using hand-crafted features only. Reason: {exc}")
        return pd.DataFrame(index=df.index)


def build_feature_frame(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    comp = pd.DataFrame([composition_features(f) for f in df["formula"]], index=df.index)
    site = pd.DataFrame([site_features(s, f) for s, f in zip(df.get("site", pd.Series(index=df.index)), df.get("facet", pd.Series(index=df.index)))], index=df.index)
    mm = _try_matminer_features(df, config)
    return pd.concat([comp, site, mm], axis=1)


def clean_features(features: pd.DataFrame) -> pd.DataFrame:
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.dropna(axis=1, how="all")
    nunique = features.nunique(dropna=True)
    features = features.loc[:, nunique > 1]
    medians = features.median(numeric_only=True)
    features = features.fillna(medians).fillna(0.0)
    return features


def _metadata(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in METADATA_COLUMNS:
        out[col] = df[col] if col in df.columns else ""
    return out


def featurize_her(her_raw: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    df = her_raw.copy()
    df["dG_H"] = pd.to_numeric(df.get("dG_H", df.get("reaction_energy")), errors="coerce")
    df = df.dropna(subset=["formula", "dG_H"])
    df = df[df["dG_H"].between(-5.0, 5.0)]
    features = clean_features(build_feature_frame(df, config))
    return pd.concat([_metadata(df).reset_index(drop=True), df[["dG_H"]].reset_index(drop=True), features.reset_index(drop=True)], axis=1)


def featurize_oer(oer_raw: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    df = oer_raw.copy()
    df["reaction_energy"] = pd.to_numeric(df["reaction_energy"], errors="coerce")
    df = df.dropna(subset=["candidate_id", "formula", "adsorbate", "reaction_energy"])
    df = df[df["reaction_energy"].between(-5.0, 8.0)]
    keys = ["candidate_id", "formula", "surface", "facet", "source", "fallback_used"]
    pivot = df.pivot_table(index=keys, columns="adsorbate", values="reaction_energy", aggfunc="mean").reset_index()
    pivot = pivot.dropna(subset=["OH", "O", "OOH"]).rename(columns={"OH": "dG_OH", "O": "dG_O", "OOH": "dG_OOH"})
    pivot["site"] = "mixed"
    # Standard four-step OER free-energy ladder at U=0 and pH=0:
    # ΔG1 = G(OH*) - G(*), ΔG2 = G(O*) - G(OH*),
    # ΔG3 = G(OOH*) - G(O*), ΔG4 = 4.92 eV - G(OOH*).
    # The theoretical overpotential in volts is max(ΔGi)/e - 1.23; because the
    # energies are in eV per electron, the numeric eV value equals volts.
    step1 = pivot["dG_OH"]
    step2 = pivot["dG_O"] - pivot["dG_OH"]
    step3 = pivot["dG_OOH"] - pivot["dG_O"]
    step4 = 4.92 - pivot["dG_OOH"]
    pivot["overpotential"] = pd.concat([step1, step2, step3, step4], axis=1).max(axis=1) - 1.23
    pivot["dG_O_minus_OH"] = pivot["dG_O"] - pivot["dG_OH"]
    features = clean_features(build_feature_frame(pivot, config))
    targets = pivot[["dG_OH", "dG_O", "dG_OOH", "overpotential", "dG_O_minus_OH"]].reset_index(drop=True)
    return pd.concat([_metadata(pivot).reset_index(drop=True), targets, features.reset_index(drop=True)], axis=1)


def featurize(config_path: str | Path = DEFAULT_CONFIG) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = load_config(config_path)
    processed_dir = Path(config["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(config["paths"]["raw_dir"])
    her_raw = pd.read_csv(raw_dir / "her_reactions.csv")
    oer_raw = pd.read_csv(raw_dir / "oer_reactions.csv")
    her = featurize_her(her_raw, config)
    oer = featurize_oer(oer_raw, config)
    her_path = processed_dir / "her_features.csv"
    oer_path = processed_dir / "oer_features.csv"
    her.to_csv(her_path, index=False)
    oer.to_csv(oer_path, index=False)
    print(f"Saved HER features: {her.shape} -> {her_path}")
    print(f"Saved OER features: {oer.shape} -> {oer_path}")
    return her, oer


def main() -> None:
    parser = argparse.ArgumentParser(description="Build feature matrices for HER and OER.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    featurize(args.config)


if __name__ == "__main__":
    main()
