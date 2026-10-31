"""Loading, cleaning and feature engineering for the Home Credit data.

Data sources, in order of preference:
  * data/raw/kaggle/application_train.csv + the six history tables (full Kaggle
    release, 307,511 applicants). History features come from src/history.py.
  * data/raw/home_credit_train.csv, a public GitHub mirror of the application table
    only (215,257 applicants, some columns renamed), fetched by scripts/download_data.sh.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
KAGGLE_DIR = DATA_DIR / "kaggle"
PROCESSED_DIR = ROOT / "data" / "processed"
HISTORY_FILE = PROCESSED_DIR / "history_features.pkl"
HISTORY_PREFIXES = ("BUR_", "PREV_", "INST_", "POS_", "CC_")
TARGET = "TARGET"
ID_COL = "SK_ID_CURR"

# The mirror renamed a few Kaggle columns. Map them back so the code (and anyone
# reading HomeCredit_columns_description.csv) sees the official names.
MIRROR_RENAMES = {
    "ID": "SK_ID_CURR",
    "GENDER": "CODE_GENDER",
    "DAYS_AGE": "DAYS_BIRTH",
    "DAYS_EMPLOYMENT": "DAYS_EMPLOYED",
}
# EXT_SOURCE_4 does not exist in the Kaggle data. It was added by whoever built the
# mirror and correlates 0.52 with EXT_SOURCE_2, so we treat it as untrusted and drop it.
DROP_COLS = ["EXT_SOURCE_4"]

DAYS_EMPLOYED_SENTINEL = 365243  # Kaggle placeholder (~1000 years), mostly pensioners


def find_raw_file() -> Path:
    for p in (KAGGLE_DIR / "application_train.csv", DATA_DIR / "application_train.csv",
              DATA_DIR / "home_credit_train.csv"):
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No data found in {DATA_DIR}. Run `bash scripts/download_data.sh` "
        "or put Kaggle's application_train.csv there."
    )


def load_raw(path: Path | None = None, history: bool = True) -> pd.DataFrame:
    """Application table, plus per-applicant history features when they have been built."""
    path = path or find_raw_file()
    df = pd.read_csv(path)
    df = df.rename(columns=MIRROR_RENAMES)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    if history and HISTORY_FILE.exists():
        df = df.join(pd.read_pickle(HISTORY_FILE), on=ID_COL)
    return df


def history_columns(cols) -> list[str]:
    return [c for c in cols if c.startswith(HISTORY_PREFIXES) or c.startswith("HIST_")]


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Clean obvious data errors and add ratio features. Row-wise only, so it is
    safe to apply before the train/test split (no statistics are learned here)."""
    df = df.copy()

    # Data errors
    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL).astype(int)
    df.loc[df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL, "DAYS_EMPLOYED"] = np.nan
    df["CODE_GENDER"] = df["CODE_GENDER"].replace("XNA", np.nan)

    # Human-readable versions of the day counts
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    df["YEARS_EMPLOYED"] = -df["DAYS_EMPLOYED"] / 365.25

    # Affordability ratios (the "ability to repay" quantities the CFPB rules talk about)
    inc = df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / inc
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / inc
    df["CREDIT_TERM"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]
    df["GOODS_CREDIT_RATIO"] = df["AMT_GOODS_PRICE"] / df["AMT_CREDIT"]
    df["EMPLOYED_AGE_RATIO"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]

    # External credit scores
    ext = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
    df["EXT_SOURCE_MEAN"] = ext.mean(axis=1)
    df["EXT_SOURCE_MIN"] = ext.min(axis=1)
    df["EXT_SOURCE_STD"] = ext.std(axis=1)
    df["EXT_SOURCE_N_MISSING"] = ext.isna().sum(axis=1)

    doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
    df["DOC_COUNT"] = df[doc_cols].sum(axis=1)

    # History features relative to this application (only when history is present)
    if "BUR_AMT_CREDIT_SUM_DEBT_SUM" in df:
        df["HIST_BUREAU_DEBT_INCOME"] = df["BUR_AMT_CREDIT_SUM_DEBT_SUM"] / inc
        df["HIST_TOTAL_ANNUITY_INCOME"] = (df["AMT_ANNUITY"] + df["BUR_AMT_ANNUITY_SUM"].fillna(0)) / inc
        df["HIST_HAS_BUREAU"] = df["BUR_SK_ID_BUREAU_COUNT"].notna().astype(int)
        df["HIST_HAS_PREVIOUS"] = df["PREV_APPROVED_COUNT"].notna().astype(int)
    return df


EXT_FEATURES = [
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "EXT_SOURCE_MEAN", "EXT_SOURCE_MIN", "EXT_SOURCE_STD", "EXT_SOURCE_N_MISSING",
]


def split_xy(df: pd.DataFrame, drop_ext: bool = False):
    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET, ID_COL])
    if drop_ext:
        X = X.drop(columns=[c for c in EXT_FEATURES if c in X.columns])
    return X, y


def categorical_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]


def to_lgbm_frame(X: pd.DataFrame, categories: dict | None = None):
    """LightGBM handles NaN and categoricals natively; just cast object columns to a
    fixed category set (learned on train) so train/test/demo share the same codes."""
    X = X.copy()
    cats = categories or {}
    for c in categorical_columns(X) if categories is None else categories:
        if categories is None:
            cats[c] = sorted(X[c].dropna().unique().tolist())
        X[c] = pd.Categorical(X[c], categories=cats[c])
    return X, cats
