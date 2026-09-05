import numpy as np
import pandas as pd
import pytest

from cervical_cancer import eda


def test_detect_outliers_iqr_basic():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 100]})
    count, lower, upper = eda.detect_outliers_iqr(df, "x")
    q1, q3 = 2.25, 4.75
    assert lower == pytest.approx(q1 - 1.5 * (q3 - q1))
    assert upper == pytest.approx(q3 + 1.5 * (q3 - q1))
    assert count == 1


def test_detect_outliers_iqr_no_outliers_and_custom_k():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
    assert eda.detect_outliers_iqr(df, "x")[0] == 0
    df2 = pd.DataFrame({"x": [1, 2, 3, 4, 5, 100]})
    assert eda.detect_outliers_iqr(df2, "x", k=100)[0] == 0


def test_detect_outliers_iqr_ignores_nan():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, np.nan, 100]})
    assert eda.detect_outliers_iqr(df, "x")[0] == 1


def test_target_correlations_excludes_target_and_sorts():
    df = pd.DataFrame(
        {
            "pos": [1, 2, 3, 4],
            "neg": [4, 3, 2, 1],
            "Biopsy": [1, 2, 3, 4],
        }
    )
    corr = eda.target_correlations(df)
    assert "Biopsy" not in corr.index
    assert corr.index.tolist() == ["pos", "neg"]
    assert corr["pos"] == pytest.approx(1.0)
    assert corr["neg"] == pytest.approx(-1.0)


def test_target_correlations_ignores_non_numeric():
    df = pd.DataFrame({"a": [1, 2, 3], "s": ["x", "y", "z"], "Biopsy": [3, 2, 1]})
    corr = eda.target_correlations(df)
    assert corr.index.tolist() == ["a"]


def test_class_balance():
    y = pd.Series([0, 0, 0, 1])
    assert eda.class_balance(y) == {"negative": 3, "positive": 1, "positive_pct": 25.0}


def test_class_balance_single_class_and_empty():
    assert eda.class_balance(pd.Series([0, 0]))["positive"] == 0
    assert eda.class_balance(pd.Series([], dtype=int)) == {"negative": 0, "positive": 0, "positive_pct": 0.0}
