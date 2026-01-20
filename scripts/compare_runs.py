import argparse
import json
import pandas as pd
from pathlib import Path
import sys


def load_metrics(run_dir):
    metrics_path = Path(run_dir) / "metrics_eval.json"
    if not metrics_path.exists():
        # Try metrics.json from training
        metrics_path = Path(run_dir) / "metrics.json"

    if not metrics_path.exists():
        return None

    with open(metrics_path, "r") as f:
        return json.load(f)


def compare_runs(run1_dir, run2_dir):
    m1 = load_metrics(run1_dir)
    m2 = load_metrics(run2_dir)

    if not m1 or not m2:
        print("Error: Could not load metrics from one or both directories.")
        return

    print(f"--- Comparison ---")
    print(f"Run 1: {run1_dir}")
    print(f"Run 2: {run2_dir}")
    print("-" * 40)

    # Check overlapping keys
    keys = set(m1.keys()) | set(m2.keys())

    for k in sorted(keys):
        v1 = m1.get(k, "N/A")
        v2 = m2.get(k, "N/A")

        # Handle lists (training curves) - just take last value
        if isinstance(v1, list):
            v1 = v1[-1]
        if isinstance(v2, list):
            v2 = v2[-1]

        diff = "N/A"
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            diff = v2 - v1

        print(f"{k:<20} | {v1:<10} | {v2:<10} | Diff: {diff}")

    print("-" * 40)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run1", help="Path to first run directory")
    parser.add_argument("run2", help="Path to second run directory")
    args = parser.parse_args()

    compare_runs(args.run1, args.run2)


if __name__ == "__main__":
    main()
