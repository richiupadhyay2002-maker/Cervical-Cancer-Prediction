import numpy as np
import pandas as pd
import pytest

from cervical_cancer import preprocessing as pp


def test_coerce_numeric_replaces_question_marks_and_casts(raw_df):
    out = pp.coerce_numeric(raw_df)
    assert out["Age"].isna().sum() == 1
    assert all(pd.api.types.is_numeric_dtype(out[c]) for c in out.columns)
    assert out["Age"].iloc[0] == 18
    # original must be untouched
    assert raw_df["Age"].iloc[2] == "?"


def test_missing_summary_only_lists_columns_with_missing(raw_df):
    df = pp.coerce_numeric(raw_df)
    summary = pp.missing_summary(df)
    assert "Biopsy" not in summary.index
    assert summary.index[0] == "STDs: Time since first diagnosis"
    assert summary.loc["STDs: Time since first diagnosis", "Percentage"] == pytest.approx(80.0)
    assert summary["Missing Count"].is_monotonic_decreasing


def test_missing_summary_empty_frame():
    assert pp.missing_summary(pd.DataFrame({"a": []})).empty


@pytest.mark.parametrize("threshold,expected", [(80.0, []), (79.0, ["STDs: Time since first diagnosis"])])
def test_drop_high_missing_threshold_is_strict(raw_df, threshold, expected):
    df = pp.coerce_numeric(raw_df)
    out, dropped = pp.drop_high_missing(df, threshold)
    assert dropped == expected
    assert all(c not in out.columns for c in dropped)


def test_drop_high_missing_empty_frame():
    df = pd.DataFrame({"a": pd.Series([], dtype=float)})
    out, dropped = pp.drop_high_missing(df)
    assert dropped == []
    assert list(out.columns) == ["a"]


def test_impute_uses_mode_mean_median_by_column_type():
    df = pd.DataFrame(
        {
            "Smokes": [0.0, 1.0, 1.0, np.nan],  # binary -> mode 1
            "Age": [10.0, 20.0, np.nan, 30.0],  # mean col -> 20
            "Number of sexual partners": [1.0, 2.0, 10.0, np.nan],  # other -> median 2
        }
    )
    out = pp.impute(df)
    assert out.isna().sum().sum() == 0
    assert out["Smokes"].iloc[3] == 1.0
    assert out["Age"].iloc[2] == pytest.approx(20.0)
    assert out["Number of sexual partners"].iloc[3] == 2.0
    assert df["Smokes"].isna().sum() == 1  # input not mutated


def test_impute_leaves_all_nan_column_alone():
    df = pd.DataFrame({"Age": [np.nan, np.nan]})
    out = pp.impute(df)
    assert out["Age"].isna().all()


def test_drop_redundant_only_drops_present_columns():
    df = pd.DataFrame({"Smokes": [0], "IUD": [1], "Age": [20]})
    out, dropped = pp.drop_redundant(df)
    assert dropped == ["Smokes", "IUD"]
    assert list(out.columns) == ["Age"]


def test_clean_end_to_end(raw_df):
    out = pp.clean(raw_df)
    assert out.isna().sum().sum() == 0
    assert "Smokes" not in out.columns
    assert "STDs: Time since first diagnosis" in out.columns  # exactly 80% is kept
    assert not out.duplicated().any()
    assert out.index.tolist() == list(range(len(out)))


def test_clean_removes_duplicates_created_by_imputation():
    df = pd.DataFrame(
        {"Age": ["20", "20", "?"], "Number of sexual partners": ["2", "2", "2"], "Biopsy": ["0", "0", "0"]}
    )
    # imputing Age with mean(20, 20) = 20 makes row 3 identical to rows 1-2
    assert len(pp.clean(df)) == 1


def test_split_features_target():
    df = pd.DataFrame({"a": [1, 2], "Biopsy": [0, 1]})
    X, y = pp.split_features_target(df)
    assert list(X.columns) == ["a"]
    assert y.tolist() == [0, 1]


def test_split_features_target_missing_target_raises():
    with pytest.raises(KeyError):
        pp.split_features_target(pd.DataFrame({"a": [1]}))


def test_prepare_train_test_scales_stratifies_and_balances(synthetic_dataset):
    X, y = synthetic_dataset
    X_tr, X_te, y_tr, y_te, scaler = pp.prepare_train_test(X, y)

    assert len(X_te) == 40 and len(y_te) == 40
    # stratified: 10% positives preserved in test split
    assert y_te.sum() == 4
    # SMOTE balanced the training set
    assert (y_tr == 0).sum() == (y_tr == 1).sum() == 144
    assert X_tr.shape == (288, 3)
    # scaler fit on unresampled train data only: train features centred near 0
    assert np.allclose(X_tr[:160].mean(axis=0), 0, atol=0.2)
    assert scaler.n_features_in_ == 3


def test_prepare_train_test_without_smote_keeps_imbalance(synthetic_dataset):
    X, y = synthetic_dataset
    X_tr, X_te, y_tr, y_te, _ = pp.prepare_train_test(X, y, smote=False)
    assert len(X_tr) == 160
    assert y_tr.sum() == 16
    assert np.allclose(X_tr.mean(axis=0), 0, atol=1e-8)
    assert np.allclose(X_tr.std(axis=0), 1, atol=1e-8)


def test_prepare_train_test_is_deterministic(synthetic_dataset):
    X, y = synthetic_dataset
    a = pp.prepare_train_test(X, y, random_state=7)
    b = pp.prepare_train_test(X, y, random_state=7)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[2], b[2])


def test_load_raw_data(tmp_path, raw_df):
    path = tmp_path / "data.csv"
    raw_df.to_csv(path, index=False)
    df = pp.load_raw_data(str(path))
    assert df.shape == raw_df.shape
    assert df["Age"].isna().sum() == 1
    assert pd.api.types.is_numeric_dtype(df["Smokes"])


def test_clean_on_real_dataset_matches_notebook(real_df):
    out = pp.clean(real_df)
    assert real_df.shape == (858, 36)
    assert out.isna().sum().sum() == 0
    dropped_high = {"STDs: Time since first diagnosis", "STDs: Time since last diagnosis"}
    assert dropped_high.isdisjoint(out.columns)
    assert set(pp.REDUNDANT_COLS).isdisjoint(out.columns)
    assert out.shape[1] == 36 - 2 - 4
    assert len(out) < 858
