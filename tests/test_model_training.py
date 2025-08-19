import os
import pickle

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from modeling.model_training import (
    classification_report_to_polars,
    evaluate_model,
    tune_and_evaluate_model,
)


@pytest.fixture(scope="module")
def small_polars_df():
    # Two classes, simple deterministic scores
    # ER: 0 0 1 1 ; y_score_1: 0.1 0.3 0.7 0.9
    return pl.DataFrame(
        {
            "ER": [0, 0, 1, 1],
            "y_score_1": [0.1, 0.3, 0.7, 0.9],
        }
    )


def test_classification_report_to_polars_parses_classes_only():
    y_true = [0, 0, 1, 1, 1, 0]
    y_pred = [0, 1, 1, 1, 1, 0]
    rep = classification_report(y_true, y_pred)

    df = classification_report_to_polars(rep)

    # Must be a Polars DataFrame with expected schema
    assert isinstance(df, pl.DataFrame)
    assert set(df.columns) == {"category", "precision", "recall", "f1-score", "count"}

    # The function is designed to include only per-class rows (e.g., "0", "1")
    cats = set(df["category"].to_list())
    assert "0" in cats and "1" in cats

    # Values should be floats and non-negative
    assert (
        (df["precision"] >= 0).all()
        and (df["recall"] >= 0).all()
        and (df["f1-score"] >= 0).all()
    )


def test_evaluate_model_metrics(small_polars_df):
    fpr, tpr, roc_auc, acc, f1, rep = evaluate_model(
        small_polars_df, target="ER", y_score="y_score_1"
    )

    # FPR/TPR arrays are non-empty and monotonic by construction of roc_curve
    assert isinstance(fpr, list) and isinstance(tpr, list)
    assert len(fpr) == len(tpr) and len(fpr) >= 2

    # AUC is in [0, 1]
    assert 0.0 <= roc_auc <= 1.0

    # With threshold 0.5 and our toy data: predictions should be [0,0,1,1] -> perfect
    assert acc == 1.0
    assert f1 == 1.0
    assert isinstance(rep, str) and "precision" in rep and "recall" in rep


def test_tune_and_evaluate_model_runs_and_saves(tmp_path, monkeypatch):
    """
    Uses LogisticRegression on a synthetic dataset.
    - Mocks plot_auc to avoid plotting side effects.
    - Writes logs and model artifact into tmp_path.
    """
    # --- Data ---
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_repeated=0,
        n_classes=2,
        random_state=42,
        shuffle=True,
    )
    X_train, X_test = X[:150], X[150:]
    y_train, y_test = y[:150], y[150:]
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)

    # --- Model and params ---
    model = LogisticRegression(solver="liblinear", max_iter=1000)
    # Keep search tiny so the test is fast
    param_dist = {"C": [0.1, 1.0, 10.0]}

    # --- Mock plot_auc to a no-op (and capture calls) ---
    calls = {}

    def _fake_plot_auc(fpr, tpr, auc_val, model_name, approach):
        calls["called"] = True
        calls["args"] = (fpr, tpr, auc_val, model_name, approach)

    monkeypatch.setattr(
        "modeling.model_training.plot_auc", _fake_plot_auc, raising=False
    )

    # --- Paths ---
    log_dir = tmp_path / "logs"
    model_dir = tmp_path / "models"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # --- Run ---
    best_model, auc_val, acc, f1, report = tune_and_evaluate_model(
        model=model,
        param_dist=param_dist,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_name="logreg_test",
        n_iter=2,  # keep fast
        log_dir=str(log_dir),
        model_dir=str(model_dir),
        approach="unit_test",
    )

    # --- Asserts ---
    # Return types/values
    from sklearn.base import BaseEstimator

    assert isinstance(best_model, BaseEstimator)
    assert 0.0 <= auc_val <= 1.0
    assert 0.0 <= acc <= 1.0
    assert 0.0 <= f1 <= 1.0
    assert isinstance(report, str) and "precision" in report

    # Model file saved
    model_path = model_dir / "logreg_test.pkl"
    assert model_path.exists(), "Best model was not saved"
    with open(model_path, "rb") as f:
        loaded = pickle.load(f)
    assert isinstance(loaded, BaseEstimator)

    # Log file created
    log_path = log_dir / "logreg_test.log"
    assert log_path.exists(), "Log file was not created"

    # Plot called
    assert calls.get("called", False) is True
    assert calls["args"][3] == "logreg_test"
