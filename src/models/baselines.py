import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

class NaiveLastValue:
    def __init__(self, horizon: int):
        self.horizon = horizon

    def predict(self, last_value: float):
        return np.array([last_value] * self.horizon)

class GBMForecaster:
    def __init__(self, horizon: int, **gbm_kwargs):
        self.horizon = horizon
        self.models = [GradientBoostingRegressor(**gbm_kwargs) for _ in range(horizon)]

    def fit(self, X, Y):
        # Y: (N, H)
        for h in range(self.horizon):
            self.models[h].fit(X, Y[:, h])

    def predict(self, X):
        preds = [m.predict(X) for m in self.models]
        return np.stack(preds, axis=1)
