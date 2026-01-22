import numpy as np
import torch

class LinearEnsemble:
    def __init__(self, num_models: int):
        self.num_models = num_models
        self.weights = np.ones(num_models) / num_models

    def fit(self, preds: np.ndarray, y_true: np.ndarray):
        # preds: (N, M, H), y_true: (N, H)
        # Simple least squares per horizon
        N, M, H = preds.shape
        W = np.zeros((M, H))
        for h in range(H):
            X = preds[:, :, h]  # (N, M)
            y = y_true[:, h]    # (N,)
            w, *_ = np.linalg.lstsq(X, y, rcond=None)
            W[:, h] = w
        self.weights = W

    def predict(self, preds: np.ndarray) -> np.ndarray:
        # preds: (N, M, H)
        N, M, H = preds.shape
        out = np.zeros((N, H))
        for h in range(H):
            out[:, h] = preds[:, :, h] @ self.weights[:, h]
        return out
