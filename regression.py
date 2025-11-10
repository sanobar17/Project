DATA_DIR = "data"
SEED = 42
TEST_START = "2022-01-01"  # time-based out-of-sample start
CALIBRATION_FRACTION = 0.2
FEATURES_NUM = [
    "income", "debt_to_income", "credit_score", "loan_amount",
    "unemployment_rate_lag1", "inflation_rate_lag1", "policy_rate_lag1",
    "GDP_growth_lag1", "house_price_index_lag1"
]
FEATURES_CAT = ["loan_type", "geographic_region", "vintage_year"]
TARGET = "default_12m"
DATE_COL = "snapshot_month"

import pandas as pd
from .config import DATA_DIR, DATE_COL

def load_borrower():
    df = pd.read_csv(f"{DATA_DIR}/borrower_panel.csv", parse_dates=[DATE_COL])
    return df

def load_macro():
    m = pd.read_csv(f"{DATA_DIR}/macro_monthly.csv", parse_dates=["month"])
    # rename to align join
    m = m.rename(columns={"month": DATE_COL})
    # create lags
    for col in ["unemployment_rate", "inflation_rate", "policy_rate", "GDP_growth", "house_price_index"]:
        m[f"{col}_lag1"] = m[col].shift(1)
    return m

def merge_data():
    b = load_borrower()
    m = load_macro()
    df = b.merge(m, on=DATE_COL, how="left")
    df = df.sort_values(DATE_COL).dropna(subset=[DATE_COL])
    return df
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from .config import FEATURES_NUM, FEATURES_CAT, TARGET, DATE_COL

def clip_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    # handle extremes
    df["debt_to_income"] = df["debt_to_income"].clip(0, 5)  # cap DTI at 500%
    df["income"] = df["income"].clip(lower=0)
    df["loan_amount"] = df["loan_amount"].clip(lower=0)
    # drop rows with missing target
    df = df.dropna(subset=[TARGET])
    # simple missing handling for numerics
    for col in FEATURES_NUM:
        df[col] = df[col].fillna(df[col].median())
    for col in FEATURES_CAT:
        df[col] = df[col].fillna("Unknown")
    return df

def build_preprocessor():
    transformers = [
        ("num", StandardScaler(), FEATURES_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01), FEATURES_CAT),
    ]
    return ColumnTransformer(transformers)
