import pandas as pd
import numpy as np

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    idx = df.index
    df["hour"] = idx.hour
    df["dayofweek"] = idx.dayofweek
    df["month"] = idx.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    return df

def add_lag_features(df: pd.DataFrame, target_col: str, lags=(1, 6, 12, 24, 48)) -> pd.DataFrame:
    df = df.copy()
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    return df

def add_rolling_features(df: pd.DataFrame, target_col: str, windows=(3, 6, 12, 24)) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        df[f"{target_col}_rollmean_{w}"] = df[target_col].rolling(w).mean()
    return df

def add_hdd_cdd(df: pd.DataFrame, temp_col: str = "temperature", base_temp: float = 18.0) -> pd.DataFrame:
    df = df.copy()
    df["HDD"] = np.maximum(0, base_temp - df[temp_col])
    df["CDD"] = np.maximum(0, df[temp_col] - base_temp)
    return df

def build_feature_matrix(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    target_col = cfg["dataset"]["target_column"]
    df = add_lag_features(df, target_col)
    df = add_rolling_features(df, target_col)
    if cfg["dataset"].get("calendar_features", True):
        df = add_calendar_features(df)
    if "temperature" in df.columns:
        df = add_hdd_cdd(df, "temperature")
    df = df.dropna()
    return df
