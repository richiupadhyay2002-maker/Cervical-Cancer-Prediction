"""Data loading and preprocessing helpers shared across the notebooks."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = "kag_risk_factors_cervical_cancer.csv"
TARGET = "Biopsy"
RANDOM_STATE = 42

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

CONTINUOUS_COLS = [
    "Age", "First sexual intercourse", "Smokes (years)",
    "Smokes (packs/year)", "Hormonal Contraceptives (years)", "IUD (years)",
]

REDUNDANT_COLS = ["Smokes", "STDs", "Hormonal Contraceptives", "IUD"]

KEY_NUMERIC_FEATURES = [
    "Age", "Number of sexual partners", "First sexual intercourse",
    "Num of pregnancies", "Smokes (years)", "Smokes (packs/year)",
]


def setup_notebook(figsize=(12, 6), dpi=100):
    """Apply the common plotting style and silence warnings."""
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = figsize
    plt.rcParams["figure.dpi"] = dpi
    warnings.filterwarnings("ignore")


def load_raw_data(path=DATA_PATH):
    """Load the CSV and replace the '?' placeholder with NaN."""
    df = pd.read_csv(path)
    df.replace("?", np.nan, inplace=True)
    return df


def convert_to_numeric(df):
    """Coerce every column to numeric in place and return the frame."""
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def missing_value_summary(df, only_missing=True):
    """Per-column missing count and percentage, sorted by count descending."""
    missing = df.isnull().sum()
    summary = pd.DataFrame({
        "Missing Count": missing,
        "Missing Percentage": missing / len(df) * 100,
    }).sort_values("Missing Count", ascending=False)
    if only_missing:
        summary = summary[summary["Missing Count"] > 0]
    return summary


def drop_high_missing_columns(df, threshold=80):
    """Drop columns whose missing percentage exceeds ``threshold``.

    Returns the list of dropped column names.
    """
    summary = missing_value_summary(df)
    cols = summary[summary["Missing Percentage"] > threshold].index.tolist()
    df.drop(columns=cols, inplace=True, errors="ignore")
    return cols


def impute_missing(df):
    """Mode for binary columns, mean for continuous, median for the rest."""
    for col in df.columns:
        if col in BINARY_COLS:
            fill = df[col].mode()[0]
        elif col in CONTINUOUS_COLS:
            fill = df[col].mean()
        else:
            fill = df[col].median()
        df[col] = df[col].fillna(fill)
    return df


def drop_redundant_columns(df, cols=REDUNDANT_COLS):
    """Drop indicator columns that duplicate their year/pack counterparts.

    Returns the list of columns actually dropped.
    """
    existing = [c for c in cols if c in df.columns]
    df.drop(columns=existing, inplace=True)
    return existing


def preprocess_pipeline(path=DATA_PATH, test_size=0.2, random_state=RANDOM_STATE):
    """Full preprocessing: load, clean, impute, split, scale and SMOTE.

    Returns ``(X_train, X_test, y_train, y_test, feature_columns)`` where
    ``X_train``/``y_train`` are the SMOTE-resampled, scaled training data.
    """
    df = convert_to_numeric(load_raw_data(path))
    drop_high_missing_columns(df)
    impute_missing(df)
    drop_redundant_columns(df)
    df.drop_duplicates(keep="first", inplace=True)

    X = df.drop(TARGET, axis=1)
    y = df[TARGET]
    feature_columns = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train, y_train = SMOTE(random_state=random_state).fit_resample(X_train, y_train)
    return X_train, X_test, y_train, y_test, feature_columns
