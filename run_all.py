"""End-to-end driver for the HER/OER catalyst screening project."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.compare import compare
from src.data_acquisition import acquire_data
from src.eda import run_eda
from src.featurize import featurize
from src.models import train_models
from src.screening import run_screening


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run acquisition -> featurize -> EDA -> models -> screening -> compare.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--force-acquire", action="store_true", help="Ignore raw CSV caches and query/regenerate data.")
    args = parser.parse_args()

    steps = [
        ("1/6 data acquisition", lambda: acquire_data(args.config, force=args.force_acquire)),
        ("2/6 featurization", lambda: featurize(args.config)),
        ("3/6 EDA", lambda: run_eda(args.config)),
        ("4/6 modeling", lambda: train_models(args.config)),
        ("5/6 screening", lambda: run_screening(args.config)),
        ("6/6 comparison", lambda: compare(args.config)),
    ]
    for label, func in steps:
        print(f"\n=== Running {label} ===", flush=True)
        func()

    print("\nPipeline complete.", flush=True)
    print("Outputs:", flush=True)
    print("- Raw data: data/raw/", flush=True)
    print("- Processed features: data/processed/", flush=True)
    print("- Figures: results/figures/", flush=True)
    print("- Metrics: results/metrics/", flush=True)
    print("- Rankings: results/rankings/", flush=True)
    print("- Comparison: results/comparison_summary.md", flush=True)


if __name__ == "__main__":
    main()
