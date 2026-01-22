import numpy as np
import pandas as pd

# Placeholder: in practice you’d plug in a proper VMD implementation
def vmd_decompose(series: pd.Series, num_modes: int = 5, **kwargs) -> pd.DataFrame:
    """
    Simple placeholder: splits the series into num_modes overlapping smoothed components.
    Replace with a real VMD implementation for production.
    """
    components = {}
    for k in range(num_modes):
        window = 3 + 2 * k
        components[f"mode_{k+1}"] = series.rolling(window, min_periods=1).mean()
    return pd.DataFrame(components, index=series.index)
