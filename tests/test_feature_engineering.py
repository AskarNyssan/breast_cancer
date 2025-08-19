import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from modeling.feature_engineering import (
    robust_scaler,
    run_tsne,
    run_umap,
    run_umap_test_set,
)


# ---------- Fixtures ----------
@pytest.fixture(scope="module")
def small_df() -> pl.DataFrame:
    rng = np.random.default_rng(42)
    n = 400
    return pl.DataFrame(
        {
            "Patient ID": np.arange(1, n + 1, dtype=np.int64),
            "f1": rng.normal(loc=100.0, scale=20.0, size=n),
            "f2": rng.exponential(scale=2.0, size=n),
            "f3": rng.normal(loc=-5.0, scale=1.5, size=n),
            "ER": rng.integers(0, 2, size=n, endpoint=False, dtype=np.int64),
        }
    )


@pytest.fixture()
def umap_params_path(tmp_path: Path) -> Path:
    # Mirror the hardcoded relative path used by the code:
    rel = Path("dataset/processed/approach_4/best_umap_params.json")
    full = tmp_path / rel
    full.parent.mkdir(parents=True, exist_ok=True)
    return full


# ---------- Tests ----------
def test_robust_scaler_keeps_id_and_target_and_centers_median(small_df: pl.DataFrame):
    out = robust_scaler(small_df, id_column="Patient ID", target_column="ER")

    # shape and column presence
    assert isinstance(out, pl.DataFrame)
    assert out.shape == small_df.shape
    assert "Patient ID" in out.columns and "ER" in out.columns

    # ID/target preserved exactly
    assert (
        out.select("Patient ID").to_series().to_list()
        == small_df.select("Patient ID").to_series().to_list()
    )
    assert (
        out.select("ER").to_series().to_list()
        == small_df.select("ER").to_series().to_list()
    )

    # check median ~ 0 for scaled feature columns
    for col in out.drop(["Patient ID", "ER"]).columns:
        med = float(out.select(pl.col(col).median()).item())
        assert abs(med) < 1e-6, (
            f"Median of {col} expected ~0 after RobustScaler, got {med}"
        )


def test_run_tsne_basic_shape_and_columns(small_df: pl.DataFrame):
    out = run_tsne(small_df, id_column="Patient ID", target_column="ER", n_components=2)

    assert out.height == small_df.height
    assert {"Patient ID", "ER", "tSNE_0", "tSNE_1"}.issubset(set(out.columns))
    # finite values
    tsne_vals = out.select(["tSNE_0", "tSNE_1"]).to_numpy()
    assert np.isfinite(tsne_vals).all()


def test_run_umap_smoke_no_save(small_df: pl.DataFrame):
    out = run_umap(
        small_df,
        id_column="Patient ID",
        target_column="ER",
        n_components=3,
        n_trials=2,  # keep tests fast
        save=False,
    )
    assert out.height == small_df.height
    assert {"Patient ID", "ER", "UMAP_0", "UMAP_1", "UMAP_2"}.issubset(set(out.columns))
    emb = out.select(["UMAP_0", "UMAP_1", "UMAP_2"]).to_numpy()
    assert np.isfinite(emb).all()


def test_run_umap_saves_best_params(
    tmp_path: Path, small_df: pl.DataFrame, monkeypatch
):
    # work in tmp so relative hardcoded path is created there
    monkeypatch.chdir(tmp_path)

    # pre-create folders that the code expects
    params_file = Path("dataset/processed/approach_4/best_umap_params.json")
    params_file.parent.mkdir(parents=True, exist_ok=True)

    _ = run_umap(
        small_df,
        id_column="Patient ID",
        target_column="ER",
        n_components=2,
        n_trials=1,  # minimal trial for speed
        save=True,
    )

    assert params_file.exists(), "Expected best UMAP params JSON to be saved"

    with params_file.open() as f:
        params = json.load(f)

    # required keys that run_umap_test_set expects
    for k in [
        "n_neighbors",
        "min_dist",
        "spread",
        "metric",
        "negative_sample_rate",
        "learning_rate",
    ]:
        assert k in params, f"Missing '{k}' in saved UMAP params"


def test_run_umap_test_set_uses_saved_params(
    tmp_path: Path, small_df: pl.DataFrame, umap_params_path: Path, monkeypatch
):
    # Prepare a valid params JSON compatible with umap.UMAP
    params = {
        "n_neighbors": 10,
        "min_dist": 0.1,
        "spread": 1.0,
        "metric": "euclidean",
        "negative_sample_rate": 5,
        "learning_rate": 1.0,
    }

    umap_params_path.write_text(json.dumps(params))
    # Run in tmp so the function's hardcoded relative path resolves
    monkeypatch.chdir(tmp_path)

    out = run_umap_test_set(
        small_df,
        id_column="Patient ID",
        target_column="ER",
        n_components=3,
    )

    # shape/columns
    assert out.height == small_df.height
    assert {"Patient ID", "ER", "UMAP_0", "UMAP_1", "UMAP_2"}.issubset(set(out.columns))
    assert np.isfinite(out.select(["UMAP_0", "UMAP_1", "UMAP_2"]).to_numpy()).all()
