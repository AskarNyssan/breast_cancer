import json
import logging
from typing import Dict, List, Literal, Optional, Set

import numpy as np
import optuna
import polars as pl
import umap.umap_ as umap
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler, StandardScaler


optuna.logging.set_verbosity(optuna.logging.WARNING)


def robust_scaler(
    data: pl.DataFrame, id_column: str = "Patient ID", target_column: str = "ER"
) -> pl.DataFrame:
    """
    Applies RobustScaler normalization to all numerical features in a Polars DataFrame,
    excluding the ID and target columns.
    Parameters:
    - data (pl.DataFrame): The input Polars DataFrame.
    - id_column (str): The name of the ID column to retain.
    - target_column (str): The name of the target column to retain.
    Returns:
    - pl.DataFrame: A new DataFrame with scaled features while keeping ID and target intact.
    """

    id_col = data.select(id_column)
    target_col = data.select(target_column)

    feature_columns = data.drop([id_column, target_column]).columns
    features = data.select(feature_columns)

    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(features.to_numpy())

    scaled_df = pl.DataFrame(scaled_features, schema=feature_columns)

    final_df = id_col.hstack(scaled_df).hstack(target_col)

    return final_df


def run_umap(
    data: pl.DataFrame,
    id_column: str = "Patient ID",
    target_column: str = "ER",
    n_components: int = 100,
    n_trials: int = 50,
    save=True,
) -> pl.DataFrame:
    """
    Applies UMAP dimensionality reduction with hyperparameter tuning using Optuna.
    Parameters:
    - data (pl.DataFrame): The input Polars DataFrame.
    - id_column (str): The name of the column containing unique IDs.
    - target_column (str): The name of the target column.
    - n_components (int): The number of UMAP components (default: 100).
    - n_trials (int): The number of UMAP iterations (default: 50).
    Returns:
    - pl.DataFrame: A DataFrame containing the ID, UMAP-reduced features, and target column.
    """

    data_ids = data.select(id_column)
    data_target = data.select(target_column)

    X_data = data.drop([id_column, target_column])

    def objective(trial) -> float:
        """
        Objective function for Optuna hyperparameter tuning.
        Parameters:
        - trial (optuna.Trial): A single Optuna trial.
        Returns:
        - float: Trustworthiness score of the UMAP transformation.
        """
        n_neighbors = trial.suggest_int("n_neighbors", 5, 100)
        min_dist = trial.suggest_float("min_dist", 0.001, 0.5, log=True)
        spread = trial.suggest_float("spread", 0.5, 2.0, log=True)
        metric = trial.suggest_categorical(
            "metric", ["euclidean", "cosine", "manhattan", "correlation"]
        )
        negative_sample_rate = trial.suggest_int("negative_sample_rate", 1, 20)
        learning_rate = trial.suggest_float("learning_rate", 0.1, 5.0, log=True)

        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            spread=spread,
            metric=metric,
            negative_sample_rate=negative_sample_rate,
            learning_rate=learning_rate,
        )

        embedding = reducer.fit_transform(X_data.to_numpy())

        trust = trustworthiness(X_data.to_numpy(), embedding, n_neighbors=n_neighbors)

        return trust

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params: Dict[str, float] = study.best_params
    if save == True:
        best_params_path = "dataset/processed/approach_4/best_umap_params.json"

        with open(best_params_path, "w") as f:
            json.dump(best_params, f)

    print("Best UMAP Parameters Saved:", best_params)

    best_reducer = umap.UMAP(n_components=n_components, **best_params)
    train_embedding = best_reducer.fit_transform(X_data.to_numpy())

    train_embedding_df = pl.DataFrame(
        train_embedding, schema=[f"UMAP_{i}" for i in range(n_components)]
    )

    train_final = pl.concat(
        [data_ids, train_embedding_df, data_target], how="horizontal"
    )

    return train_final


def run_umap_test_set(
    data: pl.DataFrame,
    id_column: str = "Patient ID",
    target_column: str = "ER",
    n_components: int = 100,
) -> pl.DataFrame:
    """
    Applies a pre-trained UMAP transformation to a test dataset using the best hyperparameters.
    Parameters:
    - data (pl.DataFrame): The input Polars DataFrame for the test set.
    - id_column (str): The column name containing unique IDs.
    - target_column (str): The column name containing the target variable.
    - n_components (int): The number of UMAP components (default: 100).
    Returns:
    - pl.DataFrame: A transformed DataFrame containing the ID, UMAP-reduced features, and target.
    """

    data_ids = data.select(id_column)
    data_target = data.select(target_column)

    X_data = data.drop([id_column, target_column])

    best_params_path = "dataset/processed/approach_4/best_umap_params.json"

    with open(best_params_path, "r") as f:
        best_params: Dict[str, float] = json.load(f)

    best_params["n_components"] = n_components

    best_reducer = umap.UMAP(**best_params)
    test_embedding = best_reducer.fit_transform(
        X_data.to_numpy()
    )  # UMAP requires NumPy array

    test_embedding_df = pl.DataFrame(
        test_embedding, schema=[f"UMAP_{i}" for i in range(n_components)]
    )

    test_final = pl.concat([data_ids, test_embedding_df, data_target], how="horizontal")

    test_trustworthiness = trustworthiness(
        X_data.to_numpy(), test_embedding, n_neighbors=best_params["n_neighbors"]
    )
    print("Test Set Trustworthiness Score:", test_trustworthiness)

    return test_final


def run_tsne(
    data: pl.DataFrame,
    id_column: str = "Patient ID",
    target_column: str = "ER",
    n_components: int = 2,
) -> pl.DataFrame:
    """
    Applies t-SNE for dimensionality reduction on a Polars DataFrame.
    Parameters:
    - data (pl.DataFrame): The input Polars DataFrame.
    - id_column (str): The column name containing unique IDs.
    - target_column (str): The column name containing the target variable.
    - n_components (int): The number of t-SNE components (default: 2).
    Returns:
    - pl.DataFrame: A DataFrame containing the ID, t-SNE reduced features, and target column.
    """

    # Extract ID and target columns before transformation
    data_ids = data.select(id_column)
    data_target = data.select(target_column)

    # Drop ID and target columns before applying t-SNE
    X_data = data.drop([id_column, target_column])

    # Apply t-SNE for dimensionality reduction
    tsne_model = TSNE(n_components=n_components, random_state=42)
    tsne_embedding = tsne_model.fit_transform(
        X_data.to_numpy()
    )  # t-SNE requires a NumPy array

    # Convert the result to a Polars DataFrame
    tsne_embedding_df = pl.DataFrame(
        tsne_embedding, schema=[f"tSNE_{i}" for i in range(n_components)]
    )

    # Concatenate ID, transformed features, and target column
    final_df = pl.concat([data_ids, tsne_embedding_df, data_target], how="horizontal")

    return final_df
