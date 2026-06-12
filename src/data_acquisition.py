"""Acquire and cache HER/OER reaction energies from Catalysis-Hub.

The extractor first tries the public GraphQL endpoint using a bounded cursor scan.
If the endpoint is unavailable or does not yield enough complete examples, it
creates a small deterministic fallback dataset so downstream ML steps remain
reproducible and runnable offline.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.yaml"

REACTION_QUERY = """
query($first:Int, $after:String) {
  reactions(first:$first, after:$after) {
    totalCount
    pageInfo { hasNextPage endCursor }
    edges {
      cursor
      node {
        id
        chemicalComposition
        surfaceComposition
        facet
        sites
        reactants
        products
        reactionEnergy
        Equation
      }
    }
  }
}
"""

ADSORBATE_KEYS = {
    "Hstar": "H",
    "OHstar": "OH",
    "HOstar": "OH",
    "Ostar": "O",
    "OOHstar": "OOH",
    "HOOstar": "OOH",
}

ELEMENT_POOL = [
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Pt",
    "Pd",
    "Ir",
    "Ru",
    "Mn",
    "Mo",
    "W",
    "Ag",
    "Au",
    "Ti",
    "V",
    "Cr",
]

SYNTHETIC_PROPS = {
    "Fe": (1.83, 26, 8),
    "Co": (1.88, 27, 9),
    "Ni": (1.91, 28, 10),
    "Cu": (1.90, 29, 11),
    "Pt": (2.28, 78, 10),
    "Pd": (2.20, 46, 10),
    "Ir": (2.20, 77, 9),
    "Ru": (2.20, 44, 8),
    "Mn": (1.55, 25, 7),
    "Mo": (2.16, 42, 6),
    "W": (2.36, 74, 6),
    "Ag": (1.93, 47, 11),
    "Au": (2.54, 79, 11),
    "Ti": (1.54, 22, 4),
    "V": (1.63, 23, 5),
    "Cr": (1.66, 24, 6),
}


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_dirs(config: dict[str, Any]) -> None:
    for key in ["raw_dir", "processed_dir", "figures_dir", "metrics_dir", "rankings_dir"]:
        Path(config["paths"][key]).mkdir(parents=True, exist_ok=True)


def _graphql_page(config: dict[str, Any], after: str | None) -> dict[str, Any]:
    data_cfg = config["data"]
    payload = {"query": REACTION_QUERY, "variables": {"first": data_cfg["page_size"], "after": after}}
    last_error: Exception | None = None
    for attempt in range(data_cfg["request_retries"]):
        try:
            response = requests.post(
                data_cfg["graphql_endpoint"],
                json=payload,
                timeout=data_cfg["request_timeout"],
            )
            response.raise_for_status()
            parsed = response.json()
            if parsed.get("errors"):
                raise RuntimeError(parsed["errors"])
            return parsed["data"]["reactions"]
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Catalysis-Hub query failed after retries: {last_error}")


def _json_loads(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value in (None, "") or (isinstance(value, float) and math.isnan(value)):
        return {}
    try:
        return json.loads(value)
    except Exception:
        return {}


def _adsorbates_from_products(products: Any) -> list[str]:
    parsed = _json_loads(products)
    adsorbates: list[str] = []
    for key in parsed:
        mapped = ADSORBATE_KEYS.get(str(key))
        if mapped and mapped not in adsorbates:
            adsorbates.append(mapped)
    return adsorbates


def _site_for_adsorbate(sites: Any, adsorbate: str) -> str | None:
    parsed = _json_loads(sites)
    for key, value in parsed.items():
        if key.upper() == adsorbate.upper() or ADSORBATE_KEYS.get(f"{key}star") == adsorbate:
            return str(value)
    return None


def _candidate_id(row: dict[str, Any]) -> str:
    formula = row.get("chemicalComposition") or "unknown"
    surface = row.get("surfaceComposition") or formula
    facet = row.get("facet") or "unknown"
    return f"{formula}|{surface}|{facet}"


def fetch_from_catalysis_hub(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bounded cursor scan of Catalysis-Hub reactions.

    The API does not expose a stable adsorbate-specific filter in all deployments,
    so the simplest robust choice is to scan a capped prefix and filter product
    JSON keys locally. This obeys the data-volume constraint in ``config.yaml``.
    """

    data_cfg = config["data"]
    her_rows: list[dict[str, Any]] = []
    oer_rows: list[dict[str, Any]] = []
    after = None
    scanned = 0

    while scanned < data_cfg["max_scan_rows"]:
        page = _graphql_page(config, after)
        edges = page.get("edges") or []
        if not edges:
            break
        for edge in edges:
            node = edge.get("node") or {}
            scanned += 1
            energy = node.get("reactionEnergy")
            formula = node.get("chemicalComposition")
            if energy is None or not formula:
                continue
            for adsorbate in _adsorbates_from_products(node.get("products")):
                base = {
                    "source_id": node.get("id"),
                    "candidate_id": _candidate_id(node),
                    "formula": formula,
                    "surface": node.get("surfaceComposition") or formula,
                    "facet": node.get("facet"),
                    "reaction": node.get("Equation"),
                    "adsorbate": adsorbate,
                    "site": _site_for_adsorbate(node.get("sites"), adsorbate),
                    "reaction_energy": float(energy),
                    "source": "Catalysis-Hub",
                    "fallback_used": False,
                }
                if adsorbate == "H" and len(her_rows) < data_cfg["her_row_cap"]:
                    row = dict(base)
                    row["dG_H"] = float(energy)
                    her_rows.append(row)
                elif adsorbate in {"OH", "O", "OOH"} and len(oer_rows) < data_cfg["oer_row_cap"]:
                    oer_rows.append(base)
        if len(her_rows) >= data_cfg["her_row_cap"] and len(oer_rows) >= data_cfg["oer_row_cap"]:
            break
        page_info = page.get("pageInfo") or {}
        if not page_info.get("hasNextPage"):
            break
        after = page_info.get("endCursor")

    return pd.DataFrame(her_rows), pd.DataFrame(oer_rows)


def _synthetic_formula(rng: np.random.Generator, with_oxygen: bool) -> str:
    n_metals = int(rng.integers(1, 4))
    metals = list(rng.choice(ELEMENT_POOL, size=n_metals, replace=False))
    counts = rng.integers(1, 5, size=n_metals)
    formula = "".join(f"{el}{count if count > 1 else ''}" for el, count in zip(metals, counts))
    if with_oxygen:
        formula += f"O{int(max(1, round(counts.sum() * rng.uniform(0.7, 1.5))))}"
    return formula


def _synthetic_signal(formula: str) -> tuple[float, float, float]:
    metals = [el for el in ELEMENT_POOL if el in formula]
    if not metals:
        metals = ["Fe"]
    vals = np.array([SYNTHETIC_PROPS[el] for el in metals], dtype=float)
    en, z, d = vals[:, 0].mean(), vals[:, 1].mean(), vals[:, 2].mean()
    return en, z, d


def fallback_datasets(config: dict[str, Any], need_her: bool, need_oer: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create small deterministic datasets when the API path is not usable.

    These values are not claimed to be DFT data; they are smooth synthetic labels
    generated from elemental trends so the course-project pipeline can be tested
    offline and the fallback is transparent in reports and cached CSVs.
    """

    rng = np.random.default_rng(config["random_seed"])
    n = int(config["data"]["fallback_rows"])
    her_rows: list[dict[str, Any]] = []
    oer_rows: list[dict[str, Any]] = []

    if need_her:
        for i in range(n):
            formula = _synthetic_formula(rng, with_oxygen=False)
            en, z, d = _synthetic_signal(formula)
            noise = rng.normal(0.0, 0.08)
            d_g_h = 0.45 * (en - 2.0) - 0.035 * (d - 8.5) + 0.002 * (z - 45) + noise
            her_rows.append(
                {
                    "source_id": f"fallback-her-{i}",
                    "candidate_id": f"{formula}|synthetic|111",
                    "formula": formula,
                    "surface": formula,
                    "facet": "111",
                    "reaction": "0.5H2(g) + * -> H*",
                    "adsorbate": "H",
                    "site": "top",
                    "reaction_energy": d_g_h,
                    "dG_H": d_g_h,
                    "source": "synthetic_fallback",
                    "fallback_used": True,
                }
            )

    if need_oer:
        for i in range(n):
            formula = _synthetic_formula(rng, with_oxygen=True)
            en, z, d = _synthetic_signal(formula)
            base = 1.05 + 0.22 * (en - 2.0) - 0.025 * (d - 8.0) + rng.normal(0.0, 0.08)
            d_g_oh = max(0.1, base)
            d_g_o = d_g_oh + 1.45 + 0.018 * (z % 10) + rng.normal(0.0, 0.10)
            d_g_ooh = d_g_oh + 3.15 + 0.10 * (2.0 - en) + rng.normal(0.0, 0.12)
            cid = f"{formula}|synthetic|110"
            for adsorbate, energy in [("OH", d_g_oh), ("O", d_g_o), ("OOH", d_g_ooh)]:
                oer_rows.append(
                    {
                        "source_id": f"fallback-oer-{i}-{adsorbate}",
                        "candidate_id": cid,
                        "formula": formula,
                        "surface": formula,
                        "facet": "110",
                        "reaction": f"synthetic {adsorbate} adsorption",
                        "adsorbate": adsorbate,
                        "site": "bridge",
                        "reaction_energy": energy,
                        "source": "synthetic_fallback",
                        "fallback_used": True,
                    }
                )

    return pd.DataFrame(her_rows), pd.DataFrame(oer_rows)


def _has_enough_oer_groups(oer_df: pd.DataFrame, min_groups: int) -> bool:
    if oer_df.empty:
        return False
    counts = oer_df.groupby("candidate_id")["adsorbate"].nunique()
    return int((counts >= 3).sum()) >= min_groups


def acquire_data(config_path: str | Path = DEFAULT_CONFIG, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = load_config(config_path)
    ensure_dirs(config)
    raw_dir = Path(config["paths"]["raw_dir"])
    her_path = raw_dir / "her_reactions.csv"
    oer_path = raw_dir / "oer_reactions.csv"

    if not force and her_path.exists() and oer_path.exists():
        print(f"Using cached raw data: {her_path}, {oer_path}")
        return pd.read_csv(her_path), pd.read_csv(oer_path)

    try:
        her_df, oer_df = fetch_from_catalysis_hub(config)
    except Exception as exc:
        print(f"WARNING: Catalysis-Hub API unavailable; using fallback datasets. Reason: {exc}")
        her_df, oer_df = pd.DataFrame(), pd.DataFrame()

    need_her = len(her_df) < int(config["data"]["min_her_rows"])
    need_oer = not _has_enough_oer_groups(oer_df, int(config["data"]["min_oer_groups"]))
    if need_her or need_oer:
        print(
            "WARNING: Catalysis-Hub returned too little complete data "
            f"(HER rows={len(her_df)}, OER rows={len(oer_df)}); using documented fallback "
            f"for {'HER' if need_her else ''} {'OER' if need_oer else ''}."
        )
        fb_her, fb_oer = fallback_datasets(config, need_her=need_her, need_oer=need_oer)
        if need_her:
            her_df = fb_her
        if need_oer:
            oer_df = fb_oer

    her_df = her_df.drop_duplicates(subset=["candidate_id", "adsorbate", "reaction_energy"]).reset_index(drop=True)
    oer_df = oer_df.drop_duplicates(subset=["candidate_id", "adsorbate", "reaction_energy"]).reset_index(drop=True)
    her_df.to_csv(her_path, index=False)
    oer_df.to_csv(oer_path, index=False)
    print(f"Saved {len(her_df)} HER rows to {her_path}")
    print(f"Saved {len(oer_df)} OER rows to {oer_path}")
    return her_df, oer_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire and cache HER/OER reaction data.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--force", action="store_true", help="Ignore existing cached CSV files.")
    args = parser.parse_args()
    acquire_data(args.config, force=args.force)


if __name__ == "__main__":
    main()
