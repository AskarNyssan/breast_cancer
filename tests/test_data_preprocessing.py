from __future__ import annotations

from typing import Dict, List, Set, Tuple

import numpy as np
import polars as pl
import pytest

from modeling.data_preprocessing import (
    convert_str_to_int,
    create_new_col_names,
    find_non_numeric,
    find_optimal_max_iter_fancyimpute,
    group_by_col_type,
    group_by_count_join_train_test,
    group_by_share_count,
    high_correlation_columns,
    join_datasets,
    ks_test_feature_selection,
    percentage_of_null_values,
    percentage_of_outliers_iforest,
    point_biserial_correlation,
    remove_outliers_isolation_forest,
    run_mice_imputation,
)


# -------------------------
# Fixtures
# -------------------------


@pytest.fixture
def df_basic() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Patient ID": [1, 2, 3, 4],
            "ER": [0, 1, 0, 1],
            "A": ["1", "2", "3", "4"],  # numeric strings
            "B": ["x", "2", "y", "4"],  # mixed strings
            "C": [10.0, 11.0, None, 14.0],
            "D": [1.0, 2.0, 3.0, 4.0],
        }
    )


@pytest.fixture
def df_join_left() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "key": [1, 2, 3],
            "L": [10, 20, 30],
        }
    )


@pytest.fixture
def df_join_right_same_type() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "key": [2, 3, 4],
            "R": [200, 300, 400],
        }
    )


@pytest.fixture
def df_join_right_castable() -> pl.DataFrame:
    # key as str so join must cast
    return pl.DataFrame(
        {
            "key": ["2", "3", "4"],
            "R": [200, 300, 400],
        }
    )


@pytest.fixture
def df_corr() -> pl.DataFrame:
    # Strongly correlated X2 ~ 2*X1, X3 uncorrelated noise
    rng = np.random.default_rng(42)
    n = 50
    x1 = rng.normal(0, 1, n)
    x2 = 2 * x1 + rng.normal(0, 1e-6, n)  # near-perfect correlation
    x3 = rng.normal(0, 1, n)
    return pl.DataFrame(
        {
            "Patient ID": np.arange(n),
            "ER": rng.integers(0, 2, n),
            "X1": x1,
            "X2": x2,
            "X3": x3,
        }
    )


@pytest.fixture
def df_bi() -> pl.DataFrame:
    # Binary ER with features; F1 separates classes, F2 doesn't
    f1 = np.r_[np.random.normal(0, 0.1, 50), np.random.normal(3, 0.1, 50)]
    f2 = np.random.normal(0, 1, 100)
    er = np.r_[np.zeros(50, dtype=int), np.ones(50, dtype=int)]
    pid = np.arange(100)
    return pl.DataFrame({"Patient ID": pid, "ER": er, "F1": f1, "F2": f2})


# -------------------------
# Tests
# -------------------------


def test_convert_str_to_int(df_basic: pl.DataFrame):
    out = convert_str_to_int(df_basic)
    assert out.schema["A"] == pl.Int64
    assert out.schema["B"] == pl.Utf8
    # Values should match integer conversion of strings
    assert out["A"].to_list() == [1, 2, 3, 4]


def test_join_datasets_same_type(
    df_join_left: pl.DataFrame, df_join_right_same_type: pl.DataFrame
):
    joined = join_datasets(
        df_join_left, df_join_right_same_type, join_col="key", method="inner"
    )
    assert joined.shape == (2, 3)
    # keys 2 and 3 intersect
    assert set(joined["key"].to_list()) == {2, 3}


def test_join_datasets_casting(df_join_left, df_join_right_castable):
    joined = join_datasets(
        df_join_left, df_join_right_castable, join_col="key", method="inner"
    )
    # Result dtype follows data_2 (right) -> string
    assert joined.schema["key"] in (pl.Utf8, pl.String)
    assert set(joined["key"].to_list()) == {"2", "3"}


def test_create_new_col_names(df_basic: pl.DataFrame):
    mapping = create_new_col_names(
        df_basic, keyword="MRI", exclude_columns=["Patient ID"]
    )
    # Should not include excluded column
    assert "Patient ID" not in mapping["MRI"].to_list()
    # All other columns should appear exactly once
    expected = [c for c in df_basic.columns if c != "Patient ID"]
    assert set(mapping["MRI"].to_list()) == set(expected)
    # New names have the right prefix
    assert all(name.startswith("MRI_") for name in mapping["New"].to_list())


def test_group_by_col_type_prints(
    df_basic: pl.DataFrame, capsys: pytest.CaptureFixture
):
    group_by_col_type(df_basic)
    captured = capsys.readouterr().out
    assert "Type" in captured and "Count" in captured


def test_find_non_numeric(df_basic: pl.DataFrame):
    # Column B contains "x" and "y" which are non-numeric
    res = find_non_numeric(df_basic, "B")
    assert res.height == 2
    assert set(res["B"].to_list()) == {"x", "y"}


def test_percentage_of_null_values_reports_sorted(df_basic: pl.DataFrame):
    res = percentage_of_null_values(df_basic)
    # Only column C has one null (out of 4 rows) -> 25%
    assert list(res.keys()) == ["C"]
    assert res["C"] == pytest.approx(25.0, abs=1e-9)


def test_group_by_share_count(df_basic: pl.DataFrame):
    out = group_by_share_count(df_basic, group_by_col="ER")
    assert out.columns == ["ER", "Count", "Percentage"]
    assert out.select(pl.col("Count").sum()).item() == df_basic.height
    assert set(out["ER"].to_list()) == {0, 1}
    assert out["Percentage"].sum() == pytest.approx(100.0)


def test_run_mice_imputation_and_optimal_iter():
    fi = pytest.importorskip("fancyimpute")
    X = np.array([[1.0, np.nan], [2.0, 2.0], [np.nan, 3.0], [4.0, 4.0]])
    imputed = run_mice_imputation(X, max_iter=2)
    assert imputed.shape == X.shape
    # Optimal iter search (low max to keep it fast)
    best_iter, imputed2 = find_optimal_max_iter_fancyimpute(
        X, tol=1e-2, max_iters=3, n_jobs=1
    )
    assert 1 <= best_iter <= 3
    assert imputed2.shape == X.shape


def test_percentage_of_outliers_iforest(df_corr: pl.DataFrame):
    out = percentage_of_outliers_iforest(
        df_corr.rename({"X1": "A1", "X2": "A2", "X3": "A3"}), target="ER"
    )
    # Expect a row per class
    assert set(out["ER"].to_list()) <= {0, 1}
    assert {"Outlier_Count", "Total_Count", "Outlier_Percentage"}.issubset(
        set(out.columns)
    )
    # Percentages are between 0 and 100
    assert (out["Outlier_Percentage"] >= 0).all() and (
        out["Outlier_Percentage"] <= 100
    ).all()


def test_high_correlation_columns(df_corr: pl.DataFrame):
    # X1 and X2 are nearly perfectly correlated; expect one of them to be marked to drop
    cols = high_correlation_columns(
        df_corr, col_to_drop=["Patient ID", "ER"], threshold=0.8
    )
    assert isinstance(cols, set)
    assert ("X1" in cols) ^ ("X2" in cols)  # exactly one is selected to drop


def test_remove_outliers_isolation_forest(df_corr: pl.DataFrame):
    df_clean, df_out = remove_outliers_isolation_forest(
        df_corr.select(["X1", "X2", "X3", "ER"]),
        contamination="auto",
        random_state=42,
        target="ER",
    )
    # All rows partitioned into clean or out
    assert df_clean.height + df_out.height == df_corr.height
    # Preserve columns
    assert set(df_clean.columns) == set(
        df_corr.select(["X1", "X2", "X3", "ER"]).columns
    )


def test_group_by_count_join_train_test():
    train = pl.DataFrame({"ER": [0, 0, 1, 1, 1], "X": [1, 2, 3, 4, 5]})
    test = pl.DataFrame({"ER": [0, 1, 1], "X": [9, 8, 7]})
    out = group_by_count_join_train_test(train, test, group_by_col="ER")
    assert out.columns == [
        "ER",
        "Train_Count",
        "Train_Percentage",
        "Test_Count",
        "Test_Percentage",
    ]
    # Counts per class
    row0 = out.filter(pl.col("ER") == 0).row(0)
    row1 = out.filter(pl.col("ER") == 1).row(0)
    # ER=0: train has 2, test has 1
    assert row0[out.columns.index("Train_Count")] == 2
    assert row0[out.columns.index("Test_Count")] == 1
    # ER=1: train has 3, test has 2
    assert row1[out.columns.index("Train_Count")] == 3
    assert row1[out.columns.index("Test_Count")] == 2


def test_point_biserial_correlation(df_bi: pl.DataFrame):
    top10, allcorr = point_biserial_correlation(
        df_bi.select(["ER", "F1", "F2"]), target="ER", top_feature_number=2
    )
    assert {"feature", "correlation"}.issubset(set(top10.columns))
    # F1 should correlate much stronger with ER than F2
    order = allcorr.sort("correlation", descending=True)["feature"].to_list()
    assert order[0] == "F1"


def test_ks_test_feature_selection(df_bi: pl.DataFrame):
    top, dropped = ks_test_feature_selection(
        df_bi.select(["Patient ID", "ER", "F1", "F2"]),
        target="ER",
        col_to_drop="Patient ID",
        threshold=0.05,
    )
    # F1 should be selected (p < 0.05), F2 likely not
    assert "F1" in top["feature"].to_list()
    assert ("F2" in dropped) or ("F2" not in top["feature"].to_list())
