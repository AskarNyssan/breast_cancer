import polars as pl
import numpy as np
from typing import Literal, List, Dict, Set
from fancyimpute import IterativeImputer
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.ensemble import IsolationForest
from scipy.stats import pointbiserialr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
from scipy.stats import ks_2samp


def convert_str_to_int(data: pl.DataFrame) -> pl.DataFrame:
    """
    Convert columns in a Polars DataFrame containing only numeric strings to integers.
    Args:
        data (pl.DataFrame): A Polars DataFrame with string columns.
    Returns:
        pl.DataFrame: A new DataFrame with applicable columns cast to Int64.
    """

    data = data.with_columns(
        [
            (data[col].cast(pl.Int64) if data[col].str.contains(r"^-?\d+$").all() else data[col])
            for col in data.columns
        ]
    )
    return data


def join_datasets(
    data_1: pl.DataFrame,
    data_2: pl.DataFrame,
    join_col: str,
    method: Literal["inner", "left", "right", "outer"] = "inner",
) -> pl.DataFrame:
    """
    Join two Polars DataFrames on a specified column.
    Args:
        data_1 (pl.DataFrame): The first DataFrame.
        data_2 (pl.DataFrame): The second DataFrame.
        join_col (str): The column name to join on.
        method (Literal["inner", "left", "right", "outer"], optional):
            The type of join to perform. Defaults to "inner".
    Returns:
        pl.DataFrame: The joined DataFrame.
    """
    if data_1.schema[join_col] == data_2.schema[join_col]:
        total_features = data_2.join(data_1, on=join_col, how=method)
    else:
        data_1 = data_1.with_columns(pl.col(join_col).cast(data_2.schema[join_col]))
        total_features = data_2.join(data_1, on=join_col, how=method)

    return total_features


def create_new_col_names(
    data: pl.DataFrame, keyword: str = "MRI", exclude_columns: List[str] = ["Patient ID"]
) -> pl.DataFrame:
    """
    Create a mapping of new column names for a Polars DataFrame by renaming
    non-excluded columns with a keyword and an index.
    Args:
        data (pl.DataFrame): The input DataFrame.
        keyword (str, optional): The prefix to use for new column names. Defaults to "MRI".
        exclude_columns (List[str], optional): List of column names to exclude. Defaults to ["Patient ID"].
    Returns:
        pl.DataFrame: A DataFrame mapping original column names to new names.
    """
    new_list: Dict[str, str] = {
        col: f"{keyword}_{i}"
        for col, i in zip(
            data.drop(exclude_columns).columns,
            range(1, len(data.drop(exclude_columns).columns) + 1),
        )
    }

    new_features_columns = pl.DataFrame(data=list(new_list.items()), schema=[f"{keyword}", "New"])

    return new_features_columns


def group_by_col_type(data: pl.DataFrame) -> None:
    """
    Group columns in a Polars DataFrame by their data type and print the count of each type.
    Args:
        data (pl.DataFrame): The input DataFrame.
    Returns:
        None: Prints the grouped DataFrame with column types and their counts.
    """
    df_types = pl.DataFrame(
        {
            "Column": data.columns,
            "Type": [str(dtype) for dtype in data.dtypes],
        }
    )

    df_grouped = df_types.group_by("Type").agg(pl.len().alias("Count"))

    print(df_grouped)


def find_non_numeric(data: pl.DataFrame, column_name: str) -> pl.DataFrame:
    """
    Find non-numeric values in a specified column of a Polars DataFrame.
    Args:
        data (pl.DataFrame): The input DataFrame.
        column_name (str): The name of the column to check for non-numeric values.
    Returns:
        pl.DataFrame: A DataFrame containing only rows where the specified column has non-numeric values.
    """
    return data.select(column_name).filter(
        ~pl.col(column_name).str.contains(r"^-?\d+(\.\d+)?$", strict=False)
    )


def percentage_of_null_values(data: pl.DataFrame) -> Dict[str, float]:
    """
    Calculate the percentage of null values for each column in a Polars DataFrame.
    Args:
        data (pl.DataFrame): The input DataFrame.
    Returns:
        Dict[str, float]: A dictionary mapping column names to their percentage of null values,
                          sorted in descending order.
    """
    null_shares = data.null_count() / data.height * 100
    df_columns = null_shares.columns
    null_share_dict: Dict[str, float] = {}

    for col in df_columns:
        filtered_series = null_shares.select(col).filter(pl.col(col) > 0).to_series().to_list()
        if len(filtered_series) == 1:
            null_share_dict[col] = np.round(filtered_series[0], 2)

    null_share_dict = dict(sorted(null_share_dict.items(), key=lambda item: item[1], reverse=True))
    if len(null_share_dict) == 0:
        print("There are no missing values in the dataset")
    else:
        return null_share_dict


def group_by_share_count(data: pl.DataFrame, group_by_col: str = "ER") -> pl.DataFrame:
    """
    Group a Polars DataFrame by a specified column, count occurrences,
    and compute the percentage share of each group.
    Args:
        data (pl.DataFrame): The input DataFrame.
        group_by_col (str, optional): The column to group by. Defaults to "ER".
    Returns:
        pl.DataFrame: A DataFrame with the computed percentage share of each group, sorted by percentage.
    """
    grouped_data = (
        data.group_by(group_by_col)
        .agg(pl.len().alias(f"Count"))
        .with_columns(
            (pl.col(f"Count") / pl.col(f"Count").sum() * 100).round(2).alias(f"Percentage")
        )
        .select([group_by_col, f"Count", f"Percentage"])
        .sort(f"Percentage")
    )

    return grouped_data


def run_mice_imputation(data: np.ndarray, max_iter: int) -> np.ndarray:
    """
    Run MICE (Multiple Imputation by Chained Equations) imputation on a dataset.
    Args:
        data (np.ndarray): The input dataset with missing values.
        max_iter (int): The number of iterations for the imputer.
    Returns:
        np.ndarray: The imputed dataset as a NumPy array.
    """
    imputer = IterativeImputer(max_iter=max_iter)
    return imputer.fit_transform(data)


def find_optimal_max_iter_fancyimpute(
    data: np.ndarray, tol: float = 1e-3, max_iters: int = 6, n_jobs: int = 6
) -> tuple[int, np.ndarray]:
    """
    Finds the optimal max_iter for MICE using fancyimpute with parallel processing.

    Args:
        data (np.ndarray): The dataset with missing values.
        tol (float, optional): The convergence threshold. Defaults to 1e-3.
        max_iters (int, optional): Maximum number of iterations to test. Defaults to 6.
        n_jobs (int, optional): Number of parallel jobs (-1 uses all available CPUs). Defaults to 6.

    Returns:
        tuple[int, np.ndarray]: (Optimal max_iter, Imputed dataset)
    """
    imputed_results = Parallel(n_jobs=n_jobs)(
        delayed(run_mice_imputation)(data, i) for i in range(1, max_iters + 1)
    )

    prev_imputed = imputed_results[0]

    for idx, imputed_data in enumerate(
        tqdm(imputed_results[1:], desc="Checking Convergence"), start=2
    ):
        diff = np.abs(imputed_data - prev_imputed).mean()
        print(f"Iteration {idx}: Mean absolute change = {diff:.6f}")

        if diff < tol:
            print(f"Converged at max_iter = {idx}")
            return idx, imputed_data

        prev_imputed = imputed_data

    print(f"⚠️ Did not converge within {max_iters} iterations. Using max_iters.")
    return max_iters, imputed_results[-1]


def percentage_of_outliers_iforest(data: pl.DataFrame, target: str = "ER") -> pl.DataFrame:
    """
    Computes the percentage of outliers for each class in the target column using Isolation Forest.
    Args:
        data (pl.DataFrame): The input DataFrame with features and a target column.
        target (str, optional): The name of the target column. Defaults to "ER".
    Returns:
        pl.DataFrame: A DataFrame containing the percentage of outliers per target class.
    """
    df_cleaned = data.drop_nulls()

    features = df_cleaned.drop(target).to_numpy()
    target_values = df_cleaned[target].to_numpy()

    iso_forest = IsolationForest(contamination="auto", random_state=42)
    outliers = iso_forest.fit_predict(features)  # -1 for outliers, 1 for normal

    df_result = pl.DataFrame({target: target_values, "Outlier": outliers})

    outlier_counts = df_result.group_by(target).agg(
        [
            (pl.col("Outlier") == -1).sum().alias("Outlier_Count"),
            pl.col("Outlier").count().alias("Total_Count"),
        ]
    )

    outlier_counts = outlier_counts.with_columns(
        (pl.col("Outlier_Count") / pl.col("Total_Count") * 100)
        .round(2)
        .alias("Outlier_Percentage")
    )

    return outlier_counts


def high_correlation_columns(
    data: pl.DataFrame, col_to_drop: List[str] = ["Patient ID", "ER"], threshold: float = 0.8
) -> Set[str]:
    """
    Identifies highly correlated columns in a Polars DataFrame.
    Args:
        data (pl.DataFrame): The input DataFrame.
        col_to_drop (List[str], optional): Columns to exclude from correlation analysis. Defaults to ["Patient ID", "ER"].
        threshold (float, optional): Correlation threshold to consider columns highly correlated. Defaults to 0.9.
    Returns:
        Set[str]: A set of column names that are highly correlated and should be dropped.
    """
    corr_matrix = data.drop(col_to_drop).corr()

    cols_to_drop: Set[str] = set()
    n_features = corr_matrix.width

    for i in range(n_features):
        for j in range(i + 1, n_features):
            if corr_matrix[i, j] > threshold:
                cols_to_drop.add(data.drop(col_to_drop).columns[j])

    return cols_to_drop


def remove_outliers_isolation_forest(
    data: pl.DataFrame, contamination="auto", random_state=42, target: str = "ER"
):
    """
    Removes outliers from a Polars DataFrame using IsolationForest.
    Parameters:
    - data: pl.DataFrame -> Input Polars DataFrame with numerical features.
    - contamination: float or 'auto' -> Proportion of outliers in the data.
    - random_state: int -> Random seed for reproducibility.
    Returns:
    - df_cleaned: pl.DataFrame -> DataFrame without outliers.
    - df_outliers: pl.DataFrame -> Outliers detected.
    """
    features = data.drop(target).to_numpy()
    target_values = data[target].to_numpy()

    iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
    outliers = iso_forest.fit_predict(features)  # -1 for outliers, 1 for normal
    predictions = pl.Series(outliers)

    df_cleaned = data.filter(predictions == 1)
    df_outliers = data.filter(predictions == -1)

    return df_cleaned, df_outliers


def group_by_count_join_train_test(
    train: pl.DataFrame, test: pl.DataFrame, group_by_col: str = "ER"
) -> pl.DataFrame:
    """
    Groups the train and test datasets by a specified column, computes count and percentage,
    and joins the results.
    Args:
        train (pl.DataFrame): The training dataset.
        test (pl.DataFrame): The testing dataset.
        group_by_col (str, optional): The column to group by. Defaults to "ER".
    Returns:
        pl.DataFrame: A dataframe with grouped counts and percentages for both train and test datasets.
    """

    train_grouped = group_by_share_count(train)
    test_grouped = group_by_share_count(test)

    data = train_grouped.join(
        test_grouped,
        on=group_by_col,
        suffix="_test",
    )

    data = data.rename(
        {
            "Count": "Train_Count",
            "Percentage": "Train_Percentage",
            "Count_test": "Test_Count",
            "Percentage_test": "Test_Percentage",
        }
    )

    return data


def point_biserial_correlation(
    data: pl.DataFrame, target: str = "ER", top_feature_number: int = 10
) -> pl.DataFrame:
    """
    Computes the Point-Biserial correlation between each numeric feature
    and a binary target variable in a given dataset.
    Args:
        data (pl.DataFrame): The input dataset containing numeric features and a binary target.
        target (str, optional): The target column for correlation computation. Defaults to "ER".
        top_feature_number (int, optional): The number of top features to return based on absolute correlation. Defaults to 10.
    Returns:
        pl.DataFrame: A dataframe containing the top features with the highest absolute correlation.
    """

    correlations: List[tuple[str, float]] = []

    for feature in data.columns:
        if feature != target:
            corr, _ = pointbiserialr(
                data[feature].to_numpy(),
                data[target].to_numpy(),
            )
            correlations.append((feature, abs(corr)))

    corr_df = pl.DataFrame(correlations, schema=["feature", "correlation"]).sort(
        "correlation", descending=True
    )

    top_features = corr_df.head(top_feature_number)

    return top_features, corr_df


def ks_test_feature_selection(
    data: pl.DataFrame,
    target: str = "ER",
    col_to_drop: str = "Patient ID",
    threshold: float = 0.05,
) -> pl.DataFrame:
    """
    Performs the Kolmogorov-Smirnov (KS) test to select features that show significant
    differences between two groups in a binary classification problem.
    Args:
        data (pl.DataFrame): The input dataset containing features and a binary target.
        target (str, optional): The binary target column. Defaults to "ER".
        col_to_drop (str, optional): Column to exclude from feature selection. Defaults to "Patient ID".
        threshold (float, optional): Significance level for feature selection. Defaults to 0.05.
    Returns:
        pl.DataFrame: A dataframe containing features with p-values below the threshold.
    """

    df_features = data.drop(target)
    df_target = data[target]

    selected_features: List[tuple[str, float]] = []
    dropped_features: List[str] = []
    for col in df_features.drop(col_to_drop).columns:
        group_0 = df_features.filter(df_target == 0).select(col).to_numpy().flatten()
        group_1 = df_features.filter(df_target == 1).select(col).to_numpy().flatten()

        p_value = ks_2samp(group_0, group_1).pvalue

        if p_value < threshold:
            selected_features.append((col, p_value))
        else:
            dropped_features.append(col)

    selected_features_df = pl.DataFrame(selected_features, schema=["feature", "p_value"])

    top_features = selected_features_df.sort("p_value", descending=False)

    return top_features, dropped_features
