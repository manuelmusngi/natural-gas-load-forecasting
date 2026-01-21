import pandas as pd
from pathlib import Path

def load_raw_timeseries(path: str) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    # Expect columns: timestamp, gas_load, weather...
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").set_index("timestamp")
    return df
