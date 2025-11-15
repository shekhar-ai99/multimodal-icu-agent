import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

# Ensure repo root is on sys.path so tests can import `src` as a package
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.eval import ShockPredictor


def test_mimic3_dummy_exists_and_columns():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "mimic3_dummy.csv")
    csv_path = os.path.abspath(csv_path)
    assert os.path.exists(csv_path), f"Expected CSV at {csv_path}"
    df = pd.read_csv(csv_path)
    expected_cols = {
        "subject_id",
        "hadm_id",
        "icu_stay_id",
        "charttime",
        "hr",
        "sysbp",
        "diasbp",
        "resp_rate",
        "spo2",
        "temp",
        "note_text",
        "cxr_path",
        "label_shock",
        "diagnosis",
    }
    assert expected_cols.issubset(set(df.columns)), "CSV missing expected columns"
    # 100 unique patients
    assert df["subject_id"].nunique() == 100, "Expected 100 unique subject_id values"


def test_shock_predictor_auc_threshold():
    """Train a simple ShockPredictor on the train split and assert AUROC > 0.85 on test."""
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    csv_path = os.path.join(base, "data", "mimic3_dummy.csv")
    meta_path = os.path.join(base, "data", "metadata.json")
    assert os.path.exists(csv_path)
    assert os.path.exists(meta_path)

    with open(meta_path) as f:
        meta = json.load(f)
    splits = meta.get("splits", {})
    train_ids = splits.get("train", [])
    test_ids = splits.get("test", [])
    assert len(train_ids) > 0 and len(test_ids) > 0

    df = pd.read_csv(csv_path)

    predictor = ShockPredictor()
    X_train, y_train = predictor.build_dataset(df, train_ids)
    X_test, y_test = predictor.build_dataset(df, test_ids)

    # Fit and evaluate
    predictor.fit(X_train, y_train)
    probs = predictor.predict_proba(X_test)
    auc = roc_auc_score(y_test, probs)
    print(f"Test AUROC: {auc:.3f}")
    assert auc > 0.85, f"AUROC too low: {auc:.3f} <= 0.85"
