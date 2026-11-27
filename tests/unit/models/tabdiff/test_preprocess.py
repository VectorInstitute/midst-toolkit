"""Unit tests for TabDiff column mapping and split membership behavior."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from midst_toolkit.common.random import set_all_random_seeds, unset_all_random_seeds


@pytest.fixture
def process_dataset(tabdiff_dir: Path):
    import process_dataset

    return process_dataset


def test_get_column_name_mapping_inverse_round_trip(process_dataset) -> None:
    columns = ["n0", "c0", "n1", "t0", "c1"]
    data_df = pd.DataFrame({name: range(3) for name in columns})
    num_col_idx = [0, 2]
    cat_col_idx = [1, 4]
    target_col_idx = [3]

    idx_mapping, inverse_idx_mapping, idx_name_mapping = process_dataset.get_column_name_mapping(
        data_df, num_col_idx, cat_col_idx, target_col_idx, columns
    )

    assert idx_name_mapping == dict(enumerate(columns))
    for original_idx, packed_idx in idx_mapping.items():
        assert inverse_idx_mapping[packed_idx] == original_idx
    # Packed layout is all numericals, then categoricals, then targets.
    assert [idx_mapping[i] for i in num_col_idx] == [0, 1]
    assert [idx_mapping[i] for i in cat_col_idx] == [2, 3]
    assert [idx_mapping[i] for i in target_col_idx] == [4]


def test_train_val_test_split_preserves_categories_and_sizes(process_dataset) -> None:
    rng = np.random.default_rng(0)
    cat_a = rng.choice(["x", "y", "z"], size=50)
    cat_b = rng.choice(["p", "q"], size=50)
    data_df = pd.DataFrame({"a": cat_a, "b": cat_b, "n": rng.normal(size=50)})
    cat_columns = ["a", "b"]

    set_all_random_seeds(999, use_deterministic_torch_algos=True, disable_torch_benchmarking=True)
    train_df, test_df, seed = process_dataset.train_val_test_split(data_df, cat_columns, num_train=40, num_test=10)
    unset_all_random_seeds()

    assert len(train_df) == 40
    assert len(test_df) == 10
    assert seed >= 1234
    for column in cat_columns:
        assert set(train_df[column]) == set(data_df[column])


def test_train_val_test_split_starts_at_seed_1234(process_dataset) -> None:
    data_df = pd.DataFrame(
        {
            "cat": ["a"] * 8 + ["b"] * 8,
            "num": np.arange(16),
        }
    )
    train_df, test_df, seed = process_dataset.train_val_test_split(data_df, ["cat"], num_train=12, num_test=4)
    assert seed == 1234
    assert len(train_df) == 12
    assert len(test_df) == 4


@pytest.mark.parametrize("fixture_key", ["berka", "diabetes"])
def test_process_data_with_test_path_does_not_resplit(process_dataset, fixture_key: str, tmp_path: Path) -> None:
    """When test_path is supplied, membership must come from the provided files."""
    import json
    import os

    from tests.unit.models.tabdiff.conftest import FIXTURE_ASSETS, load_asset_info

    asset_dir, dataset_name = FIXTURE_ASSETS[fixture_key]
    work_dir = tmp_path / "work"
    info_dir = work_dir / "data" / "Info"
    info_dir.mkdir(parents=True)
    (work_dir / "data" / dataset_name).mkdir(parents=True)

    info = load_asset_info(asset_dir, dataset_name)
    train_csv = Path(info["data_path"])
    test_csv = Path(info["test_path"])
    with open(info_dir / f"{dataset_name}.json", "w") as file:
        json.dump(info, file, indent=4)

    expected_train = pd.read_csv(train_csv)
    expected_test = pd.read_csv(test_csv)

    previous = Path.cwd()
    os.chdir(work_dir)
    try:
        process_dataset.process_data(dataset_name)
    finally:
        os.chdir(previous)

    out_train = pd.read_csv(work_dir / "data" / dataset_name / "train.csv")
    out_test = pd.read_csv(work_dir / "data" / dataset_name / "test.csv")
    assert len(out_train) == len(expected_train)
    assert len(out_test) == len(expected_test)
    # Membership identity: same rows as the supplied splits (not just same sizes).
    # process_data casts numerical columns to float32 before writing CSV.
    cols = list(expected_train.columns)
    num_columns = [cols[i] for i in info["num_col_idx"]]
    expected_train_cmp = expected_train[cols].copy()
    expected_test_cmp = expected_test[cols].copy()
    expected_train_cmp[num_columns] = expected_train_cmp[num_columns].astype(np.float32)
    expected_test_cmp[num_columns] = expected_test_cmp[num_columns].astype(np.float32)
    pd.testing.assert_frame_equal(
        out_train[cols].reset_index(drop=True),
        expected_train_cmp.reset_index(drop=True),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        out_test[cols].reset_index(drop=True),
        expected_test_cmp.reset_index(drop=True),
        check_dtype=False,
    )
    assert "?" not in expected_train.to_numpy().astype(str)
    assert "?" not in expected_test.to_numpy().astype(str)


def test_normalize_fits_quantile_on_train_only(tabdiff_dir: Path) -> None:
    """Quantile transform must fit on train; fitting on train+test changes quantiles_."""
    from sklearn.preprocessing import QuantileTransformer

    import src.data as tabdiff_data

    train = np.linspace(0.0, 1.0, 30, dtype=np.float64).reshape(-1, 1)
    # Interior points that pull quantiles if incorrectly included in fit.
    test = np.array([[0.25], [0.75]], dtype=np.float64)
    seed = 42
    n_quantiles = max(min(train.shape[0] // 30, 1000), 10)

    out, fitted = tabdiff_data.normalize(
        {"train": train.copy(), "test": test.copy()},
        "quantile",
        seed=seed,
        return_normalizer=True,
    )

    train_only = QuantileTransformer(
        output_distribution="normal",
        n_quantiles=n_quantiles,
        subsample=int(1e9),
        random_state=seed,
    )
    train_only.fit(train)
    np.testing.assert_allclose(fitted.quantiles_, train_only.quantiles_, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(out["test"], train_only.transform(test), rtol=1e-6, atol=1e-6)

    concat_fit = QuantileTransformer(
        output_distribution="normal",
        n_quantiles=n_quantiles,
        subsample=int(1e9),
        random_state=seed,
    )
    concat_fit.fit(np.vstack([train, test]))
    assert not np.allclose(concat_fit.quantiles_, fitted.quantiles_)


def test_quantile_inverse_round_trip_within_tolerance(tabdiff_dir: Path) -> None:
    """Quantile transform is not bit-exact invertible; recover within a documented tolerance."""
    import src.data as tabdiff_data

    rng = np.random.default_rng(0)
    train = rng.normal(size=(60, 2))
    test = rng.normal(size=(10, 2))
    out, normalizer = tabdiff_data.normalize(
        {"train": train.copy(), "test": test.copy()},
        "quantile",
        seed=0,
        return_normalizer=True,
    )
    recovered_train = normalizer.inverse_transform(out["train"])
    recovered_test = normalizer.inverse_transform(out["test"])
    # sklearn QuantileTransformer inverse is approximate; use a loose but fixed contract.
    np.testing.assert_allclose(recovered_train, train, rtol=0.15, atol=0.25)
    np.testing.assert_allclose(recovered_test, test, rtol=0.15, atol=0.25)


def test_cat_encode_fits_on_train_unknown_gets_extra_code(tabdiff_dir: Path) -> None:
    """Ordinal encoding fits on train; unseen test categories become max(train)+1."""
    import src.data as tabdiff_data

    X = {
        "train": np.array([["a"], ["b"], ["a"]], dtype=object),
        "test": np.array([["a"], ["c"]], dtype=object),
    }
    encoded, is_numeric = tabdiff_data.cat_encode(X, encoding=None, y_train=None, seed=0)
    assert is_numeric is False
    assert set(encoded["train"].reshape(-1).tolist()) <= {0, 1}
    train_max = int(encoded["train"].max())
    assert encoded["test"][0, 0] in (0, 1)
    assert int(encoded["test"][1, 0]) == train_max + 1


def test_prefitted_label_encoder_pickle_keeps_codes(tabdiff_dir: Path, asset_dir: Path) -> None:
    """Loaded LabelEncoder pickle keyed by column name: transform keeps codes (1 -> 1)."""
    import src.data as tabdiff_data

    names = ["trans_type", "operation", "k_symbol", "bank"]
    encoders = tabdiff_data.load_label_encoders_pickle(
        str(asset_dir / "trans_label_encoders.pkl"),
        names,
    )
    assert [len(encoders[n].classes_) for n in names] == [3, 6, 9, 14]
    X = {
        "train": np.array([[1, 2, 3, 4], [0, 5, 8, 13]], dtype=np.int64),
        "test": np.array([[2, 0, 1, 7]], dtype=np.int64),
        "val": np.zeros((0, 4), dtype=np.int64),
    }
    encoded = tabdiff_data.apply_prefitted_label_encoders(
        {k: v.copy() for k, v in X.items()},
        encoders,
        names,
    )
    np.testing.assert_array_equal(encoded["train"], X["train"])
    np.testing.assert_array_equal(encoded["test"], X["test"])


def test_diabetes_test_pickle_densifies_sparse_codes(tabdiff_dir: Path, diabetes_asset_dir: Path) -> None:
    """Fixture-local diabetes vocab is a sparse mother subset; transform densifies, inverse restores."""
    import src.data as tabdiff_data

    # weight classes_ are [1, 3, 8, 9] in diabetes_test.pkl → dense 0..3
    names = ["weight_categorical", "age_categorical"]
    encoders = tabdiff_data.load_label_encoders_pickle(
        str(diabetes_asset_dir / "diabetes_test.pkl"),
        names,
    )
    assert list(map(int, encoders["weight_categorical"].classes_)) == [1, 3, 8, 9]
    assert list(map(int, encoders["age_categorical"].classes_))[0] == 1

    X = {
        "train": np.array([[1, 1], [3, 5], [8, 9]], dtype=np.int64),
        "test": np.array([[9, 2]], dtype=np.int64),
    }
    encoded = tabdiff_data.apply_prefitted_label_encoders(
        {k: v.copy() for k, v in X.items()},
        encoders,
        names,
    )
    np.testing.assert_array_equal(encoded["train"][:, 0], [0, 1, 2])
    np.testing.assert_array_equal(encoded["test"][:, 0], [3])
    assert not np.array_equal(encoded["train"], X["train"])

    # Inverse (as used at sample time) restores original mother-space codes.
    for split in ("train", "test"):
        recovered = np.stack(
            [encoders[name].inverse_transform(encoded[split][:, i].astype(int)) for i, name in enumerate(names)],
            axis=1,
        )
        np.testing.assert_array_equal(recovered, X[split])


def test_category_encoder_kwargs_resolves_relative_pickle_path(
    tabdiff_dir: Path, asset_dir: Path, tmp_path: Path
) -> None:
    """Relative label_encoders_path is resolved against dataset_path or its parent."""
    import shutil

    import utils_train

    dataset_dir = tmp_path / "data" / "trans"
    dataset_dir.mkdir(parents=True)
    pickle_name = "trans_label_encoders.pkl"
    # Parent-of-dataset layout (Info-relative path copied beside data/<name>/).
    shutil.copy(asset_dir / pickle_name, tmp_path / "data" / pickle_name)

    info = {
        "label_encoders_path": pickle_name,
        "column_names": ["trans_date", "trans_type", "operation", "amount", "balance", "k_symbol", "bank", "account"],
        "cat_col_idx": [1, 2, 5, 6],
    }
    kwargs = utils_train._category_encoder_kwargs_from_info(info, str(dataset_dir))
    assert Path(kwargs["label_encoders_path"]).resolve() == (tmp_path / "data" / pickle_name).resolve()
    assert kwargs["categorical_column_names"] == ("trans_type", "operation", "k_symbol", "bank")

    # Inside-dataset layout wins when present.
    shutil.copy(asset_dir / pickle_name, dataset_dir / pickle_name)
    kwargs_inner = utils_train._category_encoder_kwargs_from_info(info, str(dataset_dir))
    assert Path(kwargs_inner["label_encoders_path"]).resolve() == (dataset_dir / pickle_name).resolve()


def test_category_encoder_kwargs_missing_relative_falls_back_then_load_fails(
    tabdiff_dir: Path, tmp_path: Path
) -> None:
    import utils_train

    import src.data as tabdiff_data

    dataset_dir = tmp_path / "data" / "trans"
    dataset_dir.mkdir(parents=True)
    info = {
        "label_encoders_path": "missing_encoders.pkl",
        "column_names": ["a", "b"],
        "cat_col_idx": [1],
    }
    kwargs = utils_train._category_encoder_kwargs_from_info(info, str(dataset_dir))
    assert kwargs["label_encoders_path"] == str(dataset_dir / "missing_encoders.pkl")
    with pytest.raises(FileNotFoundError):
        tabdiff_data.load_label_encoders_pickle(kwargs["label_encoders_path"], ["b"])


def test_require_supplied_splits_rejects_missing_test_path(process_dataset, asset_dir: Path, tmp_path: Path) -> None:
    import json
    import os

    from tests.unit.models.tabdiff.conftest import load_asset_info

    work_dir = tmp_path / "work"
    info_dir = work_dir / "data" / "Info"
    info_dir.mkdir(parents=True)
    (work_dir / "data" / "trans").mkdir(parents=True)
    info = load_asset_info(asset_dir)
    info["test_path"] = ""
    info["require_supplied_splits"] = True
    with open(info_dir / "trans.json", "w") as file:
        json.dump(info, file, indent=4)
    previous = Path.cwd()
    os.chdir(work_dir)
    try:
        with pytest.raises(ValueError, match="require_supplied_splits"):
            process_dataset.process_data("trans")
    finally:
        os.chdir(previous)


def test_load_label_encoders_pickle_missing_file(tabdiff_dir: Path, tmp_path: Path) -> None:
    import src.data as tabdiff_data

    with pytest.raises(FileNotFoundError, match="label_encoders_path does not exist"):
        tabdiff_data.load_label_encoders_pickle(str(tmp_path / "missing.pkl"), ["bank"])


def test_load_label_encoders_pickle_rejects_non_dict(tabdiff_dir: Path, tmp_path: Path) -> None:
    import pickle

    import src.data as tabdiff_data

    path = tmp_path / "not_a_dict.pkl"
    with path.open("wb") as handle:
        pickle.dump(["not", "a", "dict"], handle)
    with pytest.raises(ValueError, match="must unpickle to a dict"):
        tabdiff_data.load_label_encoders_pickle(str(path), ["bank"])


def test_load_label_encoders_pickle_missing_column_key(tabdiff_dir: Path, asset_dir: Path) -> None:
    import src.data as tabdiff_data

    with pytest.raises(ValueError, match="missing encoders for columns"):
        tabdiff_data.load_label_encoders_pickle(
            str(asset_dir / "trans_label_encoders.pkl"),
            ["trans_type", "not_a_real_column"],
        )
