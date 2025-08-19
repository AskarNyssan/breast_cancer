import os
import logging
from typing import Dict, Tuple, Any
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from modeling.visualisation import plot_auc
import pickle
from typing import Tuple, List
import polars as pl
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold


def tune_and_evaluate_model(
    model: BaseEstimator,
    param_dist: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    n_iter: int = 50,
    log_dir: str = "models/logs/approach_1",
    model_dir: str = "models/saved_models/approach_1",
    approach: str = "approach_1",
) -> Tuple[BaseEstimator, float, float, float]:
    """
    Tunes hyperparameters using RandomizedSearchCV and evaluates a model.
    Parameters:
    - model (BaseEstimator): The machine learning model to be tuned.
    - param_dist (Dict[str, Any]): Dictionary containing hyperparameter distributions for RandomizedSearchCV.
    - X_train (pd.DataFrame): Training feature set.
    - y_train (pd.Series): Training labels.
    - X_test (pd.DataFrame): Test feature set.
    - y_test (pd.Series): Test labels.
    - model_name (str): The name of the model for logging and display purposes.
    - n_iter (int, optional): Number of iterations for RandomizedSearchCV (default: 50).
    - log_dir (str, optional): Directory to store log files (default: "../models/logs/approach_1").
    - model_dir (str, optional): Directory to store model files (default: "../models/saved_models/approach_1").
    Returns:
    - Tuple[BaseEstimator, float, float, float]: The best model, AUC score, accuracy score, and F1-score.
    """

    # 1. Ensure the log directory exists
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"{model_name}.log")

    # 2. Remove any existing handlers so logging.basicConfig() can reconfigure properly
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # 3. Configure logging to write WARNING (and above) to `model_name.log`
    logging.basicConfig(
        filename=log_file,
        filemode="w",
        level=logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # 4. Capture all Python warnings (including scikit-learn warnings) and direct them to the logger.
    logging.captureWarnings(True)

    # 56. Set up 5-fold cross-validation
    # kf = KFold(n_splits=5, shuffle=True, random_state=42)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 6. Use RandomizedSearchCV to find the best hyperparameters
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=kf,
        n_jobs=-1,
        verbose=1,
        random_state=42,
    )
    random_search.fit(X_train, y_train)

    # 7. Get the best model
    best_model: BaseEstimator = random_search.best_estimator_

    # 8. Evaluate the best model on the test set
    y_pred_proba: np.ndarray = best_model.predict_proba(X_test)[:, 1]
    y_pred: np.ndarray = best_model.predict(X_test)

    # 9. Calculate metrics
    auc: float = roc_auc_score(y_test, y_pred_proba)
    accuracy: float = accuracy_score(y_test, y_pred)
    f1: float = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # 10. Print best hyperparameters and metrics
    print(f"{model_name} Best Hyperparameters: {random_search.best_params_}")
    print(f"{model_name} Test Metrics:")
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # 11. Plot AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plot_auc(fpr, tpr, auc, model_name, approach)

    # 12. Save the best model
    with open(f"{model_dir}/{model_name}.pkl", "wb") as f:
        pickle.dump(best_model, f)

    return best_model, auc, accuracy, f1, report


def evaluate_model(
    data: pl.DataFrame, target: str = "ER", y_score: str = "y_score_1"
) -> Tuple[List[float], List[float], float, float, float]:
    """
    Evaluates a model's performance using AUC, accuracy, and F1-score.
    Parameters:
    - data (pl.DataFrame): A Polars DataFrame.
    - target (str): target column name.
    - y_score (str): y_score column name.
    Returns:
    - Tuple[List[float], List[float], float, float, float]:
      - List of false positive rates (FPR).
      - List of true positive rates (TPR).
      - AUC score.
      - Accuracy score.
      - F1-score.
    """

    # Extract the relevant columns as lists
    y_true: List[int] = data[target].to_list()
    y_score: List[float] = data[y_score].to_list()

    # Calculate FPR, TPR, and thresholds
    fpr, tpr, _ = roc_curve(y_true, y_score)

    # Calculate AUC
    roc_auc: float = auc(fpr, tpr)

    # Calculate Accuracy
    y_pred: List[int] = [1 if score >= 0.5 else 0 for score in y_score]
    accuracy: float = accuracy_score(y_true, y_pred)

    report = classification_report(y_true, y_pred)

    # Calculate F1-score
    f1: float = f1_score(y_true, y_pred)

    return fpr.tolist(), tpr.tolist(), roc_auc, accuracy, f1, report


def classification_report_to_polars(report_str: str) -> pl.DataFrame:
    """
    Converts a classification report string into a Polars DataFrame.
    Parameters:
        report_str (str): The classification report as a string.
    Returns:
        pl.DataFrame: A structured Polars DataFrame containing precision, recall, f1-score, and support values.
    """
    # Splitting the lines and cleaning up spaces
    lines = report_str.split("\n")

    # Extracting valid data lines
    data = []
    for line in lines[2:]:  # Skipping the first two lines (header)
        parts = line.strip().split()
        if len(parts) == 5:
            category, precision, recall, f1_score, support = parts
        elif len(parts) == 4:
            category, precision, recall, f1_score = parts
            support = None
        else:
            continue
        data.append(
            [
                category,
                float(precision),
                float(recall),
                float(f1_score),
                int(support) if support else None,
            ]
        )

    # Creating the Polars DataFrame
    df = pl.DataFrame(
        data,
        schema=[
            ("category", pl.Utf8),
            ("precision", pl.Float64),
            ("recall", pl.Float64),
            ("f1-score", pl.Float64),
            ("count", pl.Int64),
        ],
    )

    return df
