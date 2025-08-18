import polars as pl
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pickle
import numpy as np
from scipy.stats import chi2_contingency
from typing import Dict, List, Any


def compute_metrics(df: pl.DataFrame) -> Dict[str, float]:
    """
    Compute accuracy, precision, recall, F1, and AUC given binary columns:
      - df["ER"] as ground-truth label (0 or 1)
      - df["ER_pred_label"] as predicted label (0 or 1)
      - df["ER_pred_proba"] as predicted probability (between 0 and 1)
    Returns a dictionary of metrics.
    """
    tp: int = df.filter((pl.col("ER") == 1) & (pl.col("ER_pred_label") == 1)).height
    tn: int = df.filter((pl.col("ER") == 0) & (pl.col("ER_pred_label") == 0)).height
    fp: int = df.filter((pl.col("ER") == 0) & (pl.col("ER_pred_label") == 1)).height
    fn: int = df.filter((pl.col("ER") == 1) & (pl.col("ER_pred_label") == 0)).height

    accuracy: float = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision: float = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall: float = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1: float = (
        (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    )

    # Compute AUC (Requires predicted probabilities)
    auc = None
    if "ER_pred_proba" in df.columns:
        y_true = df["ER"].to_list()
        y_pred_prob = df["ER_pred_proba"].to_list()
        if len(set(y_true)) > 1:
            auc = roc_auc_score(y_true, y_pred_prob)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "auc": auc if auc is not None else 0.0,  # Default to 0.0 if AUC cannot be computed
    }


def get_metrics_manufacture(
    data: pl.DataFrame, col_manufacture: str = "Manufacturer"
) -> pl.DataFrame:
    """
    Compute classification metrics (including AUC) for each manufacturer.
    Parameters:
    - data (pl.DataFrame): Input dataset containing predictions and ground truth labels.
    - col_manufacture (str): Column name representing manufacturer types.
    Returns:
    - pl.DataFrame: A DataFrame containing classification metrics for each manufacturer.
    """
    # 1. Get unique manufacturer types
    unique_manufacturers: List[str] = (
        data.select(pl.col(col_manufacture)).unique().to_series().to_list()
    )

    # 2. Initialize a list to store results for each manufacturer
    results_list: List[Dict[str, float]] = []

    # 3. Loop over each manufacturer and compute metrics
    for mfg_type in unique_manufacturers:
        # Filter to only rows for the given manufacturer
        df_sub: pl.DataFrame = data.filter(pl.col(col_manufacture) == mfg_type)
        metrics: Dict[str, float] = compute_metrics(df_sub)

        # Build a dictionary of results
        results_dict: Dict[str, float] = {
            col_manufacture: mfg_type,
            "#Samples": df_sub.height,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1-score": metrics["f1"],
            "TP": metrics["tp"],
            "FP": metrics["fp"],
            "FN": metrics["fn"],
            "TN": metrics["tn"],
            "AUC": metrics["auc"],
        }

        # Add to our list
        results_list.append(results_dict)

    # 4. Convert the list of dictionaries into a Polars DataFrame
    metrics_df: pl.DataFrame = pl.DataFrame(results_list)

    return metrics_df
