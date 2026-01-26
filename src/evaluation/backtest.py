import numpy as np
import pandas as pd
from .metrics import mae, rmse, mape

def rolling_backtest(series, model_fn, input_length, horizon, step=24):
    """
    model_fn: function that takes history array and returns forecast (H,)
    """
    idx = series.index
    values = series.values
    forecasts = []
    actuals = []
    times = []

    t = input_length
    while t + horizon <= len(values):
        hist = values[t - input_length : t]
        y_true = values[t : t + horizon]
        y_pred = model_fn(hist)
        forecasts.append(y_pred)
        actuals.append(y_true)
        times.append(idx[t])
        t += step

    forecasts = np.stack(forecasts)
    actuals = np.stack(actuals)
    metrics = {
        "MAE": mae(actuals, forecasts),
        "RMSE": rmse(actuals, forecasts),
        "MAPE": mape(actuals, forecasts),
    }
    return pd.DataFrame(metrics, index=["backtest"]), forecasts, actuals, times
