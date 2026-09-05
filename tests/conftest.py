from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cervical_cancer.preprocessing import BINARY_COLS, MEAN_COLS

REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = REPO_ROOT / "kag_risk_factors_cervical_cancer.csv"


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Small string-typed frame mimicking the Kaggle CSV before cleaning."""
    return pd.DataFrame(
        {
            "Age": ["18", "25", "?", "40", "25"],
            "Number of sexual partners": ["1", "?", "3", "4", "?"],
            "Smokes": ["0", "1", "?", "0", "1"],
            "Smokes (years)": ["0", "10", "?", "0", "10"],
            "STDs: Time since first diagnosis": ["?", "?", "?", "?", "1"],
            "Biopsy": ["0", "1", "0", "0", "1"],
        }
    )


@pytest.fixture
def synthetic_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Numeric, imbalanced dataset large enough for stratified split + SMOTE."""
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(
        {
            "Age": rng.integers(15, 60, n),
            "Number of sexual partners": rng.integers(1, 6, n),
            "Smokes (years)": rng.random(n) * 10,
        }
    )
    y = pd.Series(np.r_[np.ones(20, dtype=int), np.zeros(n - 20, dtype=int)], name="Biopsy")
    return X, y


@pytest.fixture(scope="session")
def real_df() -> pd.DataFrame:
    if not CSV_PATH.exists():
        pytest.skip("dataset CSV not present")
    return pd.read_csv(CSV_PATH)


def is_binary_col(col: str) -> bool:
    return col in BINARY_COLS


def is_mean_col(col: str) -> bool:
    return col in MEAN_COLS
