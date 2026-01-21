import pandas as pd

def filter_date_range(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df.loc[start:end].copy()

def train_val_test_split(df: pd.DataFrame, cfg: dict):
    train = filter_date_range(df, cfg["splits"]["train_start"], cfg["splits"]["train_end"])
    val = filter_date_range(df, cfg["splits"]["val_start"], cfg["splits"]["val_end"])
    test = filter_date_range(df, cfg["splits"]["test_start"], cfg["splits"]["test_end"])
    return train, val, test
