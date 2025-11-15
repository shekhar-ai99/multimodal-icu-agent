#!/usr/bin/env python3
"""
Evaluation runner: loads the dummy dataset, trains ShockPredictor on train split,
and runs full evaluation (RAGAS, hallucination, AUROC, clinician score) on test set.

Usage:
    python run_evaluation.py

Output:
    - results/metrics.json (JSON metrics)
    - results/table_1.txt (LaTeX table)
"""
import os
import sys
import json

# Ensure src is importable
ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.eval import ShockPredictor, FullEvaluator


def main():
    # paths
    csv_path = os.path.join(ROOT, "data", "mimic3_dummy.csv")
    meta_path = os.path.join(ROOT, "data", "metadata.json")
    results_dir = os.path.join(ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print("Running Full Evaluation on Dummy MIMIC-III Dataset")
    print("=" * 60)

    # 1. Load metadata to get train/val/test splits
    with open(meta_path) as f:
        meta = json.load(f)
    splits = meta.get("splits", {})
    train_ids = splits.get("train", [])
    test_ids = splits.get("test", [])
    print(f"Train subjects: {len(train_ids)}")
    print(f"Test subjects: {len(test_ids)}")

    # 2. Train ShockPredictor on train split
    print("\nTraining ShockPredictor...")
    predictor = ShockPredictor()
    X_train, y_train = predictor.fit_from_csv(csv_path, train_ids)
    print(f"Trained on {len(X_train)} training subjects")

    # 3. Run full evaluation
    print("\nRunning evaluation on test set...")
    evaluator = FullEvaluator()
    metrics = evaluator.evaluate_on_test_set(
        csv_path=csv_path,
        metadata_path=meta_path,
        agent_fn=None,  # no LLM agent in this run (would need instantiation)
        predictor=predictor,
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for key, val in sorted(metrics.items()):
        if isinstance(val, float):
            print(f"{key:30s}: {val:7.4f}")
        else:
            print(f"{key:30s}: {val}")

    # 4. Save metrics to JSON
    metrics_file = os.path.join(results_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Metrics saved to {metrics_file}")

    # 5. Generate LaTeX table
    latex_table = FullEvaluator.metrics_to_latex_table(metrics)
    table_file = os.path.join(results_dir, "table_1.txt")
    with open(table_file, "w") as f:
        f.write(latex_table)
    print(f"✓ LaTeX table saved to {table_file}")

    print("\n" + latex_table)


if __name__ == "__main__":
    main()
