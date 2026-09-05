"""Data cleaning and feature-engineering steps from ``feature_engineering.ipynb``."""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TARGET = "Biopsy"

BINARY_COLS = [
    "Smokes", "Hormonal Contraceptives", "IUD", "STDs",
    "STDs:condylomatosis", "STDs:cervical condylomatosis",
    "STDs:vaginal condylomatosis", "STDs:vulvo-perineal condylomatosis",
    "STDs:syphilis", "STDs:pelvic inflammatory disease",
    "STDs:genital herpes", "STDs:molluscum contagiosum",
    "STDs:AIDS", "STDs:HIV", "STDs:Hepatitis B", "STDs:HPV",
    "Dx:Cancer", "Dx:CIN", "Dx:HPV", "Dx",
    "Hinselmann", "Schiller", "Citology", "Biopsy",
]

MEAN_COLS = [
    "Age", "First sexual intercourse", "Smokes (years)",
    "Smokes (packs/year)", "Hormonal Contraceptives (years)", "IUD (years)",
]

REDUNDANT_COLS = ["Smokes", "STDs", "Hormonal Contraceptives", "IUD"]


def load_raw_data(path: str) -> pd.DataFrame:
    """Read the Kaggle CSV, treating ``?`` as missing and coercing to numeric."""
    df = pd.read_csv(path)
    return coerce_numeric(df)


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Replace ``?`` with NaN and convert every column to numeric."""
    out = df.replace("?", np.nan)
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missing count/percentage for columns with any missing values."""
    missing = df.isnull().sum()
    pct = missing / len(df) * 100 if len(df) else missing.astype(float)
    info = pd.DataFrame({"Missing Count": missing, "Percentage": pct})
    return info[info["Missing Count"] > 0].sort_values("Missing Count", ascending=False)


def drop_high_missing(df: pd.DataFrame, threshold: float = 80.0) -> tuple[pd.DataFrame, list[str]]:
    """Drop columns whose missing percentage exceeds ``threshold``."""
    if len(df) == 0:
        return df.copy(), []
    pct = df.isnull().sum() / len(df) * 100
    cols = pct[pct > threshold].index.tolist()
    return df.drop(columns=cols), cols


def impute(df: pd.DataFrame) -> pd.DataFrame:
    """Mode for binary columns, mean for continuous columns, median otherwise."""
    out = df.copy()
    for col in out.columns:
        if out[col].isnull().all():
            continue
        if col in BINARY_COLS:
            fill = out[col].mode()[0]
        elif col in MEAN_COLS:
            fill = out[col].mean()
        else:
            fill = out[col].median()
        out[col] = out[col].fillna(fill)
    return out


def drop_redundant(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop indicator columns whose information is carried by year/count columns."""
    cols = [c for c in REDUNDANT_COLS if c in df.columns]
    return df.drop(columns=cols), cols


def clean(df: pd.DataFrame, threshold: float = 80.0) -> pd.DataFrame:
    """Full cleaning pipeline: coerce, drop high-missing, impute, drop redundant, dedupe."""
    out = coerce_numeric(df)
    out, _ = drop_high_missing(out, threshold)
    out = impute(out)
    out, _ = drop_redundant(out)
    return out.drop_duplicates(keep="first").reset_index(drop=True)


def split_features_target(df: pd.DataFrame, target: str = TARGET) -> tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise KeyError(f"target column {target!r} not in dataframe")
    return df.drop(columns=[target]), df[target]


def prepare_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    smote: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Stratified split, scale (fit on train only), optionally SMOTE the train set."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    y_train = np.asarray(y_train)
    if smote:
        X_train_s, y_train = SMOTE(random_state=random_state).fit_resample(X_train_s, y_train)
    return X_train_s, X_test_s, np.asarray(y_train), np.asarray(y_test), scaler
