from pathlib import Path
import pandas as pd
import torch

from src.data.load_data import load_raw_timeseries
from src.data.preprocess import train_val_test_split
from src.data.feature_engineering import build_feature_matrix
from src.models.lstm_enhanced import EnhancedLSTMForecaster
from src.training.train_lstm import SequenceDataset, train_lstm_model
from src.training.train_hybrid import train_hybrid_model
from src.evaluation.metrics import mae, rmse, mape
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ForecastingPipeline:
    def __init__(self, data_config: dict, model_config: dict, training_config: dict):
        self.data_cfg = data_config
        self.model_cfg = model_config
        self.train_cfg = training_config

    def run(self):
        raw_path = Path(self.data_cfg["data_paths"]["raw"]) / "gas_load.csv"
        df = load_raw_timeseries(str(raw_path))

        df_feat = build_feature_matrix(df, self.data_cfg)
        train_df, val_df, test_df = train_val_test_split(df_feat, self.data_cfg)

        target_col = self.data_cfg["dataset"]["target_column"]
        X_train = train_df.drop(columns=[target_col]).values
        y_train = train_df[target_col].values
        X_val = val_df.drop(columns=[target_col]).values
        y_val = val_df[target_col].values
        X_test = test_df.drop(columns=[target_col]).values
        y_test = test_df[target_col].values

        # Enhanced LSTM
        lstm_cfg = self.model_cfg["lstm_enhanced"]
        input_length = lstm_cfg["input_length"]
        horizon = lstm_cfg["forecast_horizon"]

        train_dataset = SequenceDataset(X_train, y_train, input_length, horizon)
        val_dataset = SequenceDataset(X_val, y_val, input_length, horizon)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.train_cfg["training"]["batch_size"],
            shuffle=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.train_cfg["training"]["batch_size"],
            shuffle=False,
        )

        model_lstm = EnhancedLSTMForecaster(
            input_size=X_train.shape[1],
            forecast_horizon=horizon,
            hidden_size=lstm_cfg["hidden_size"],
            num_layers=lstm_cfg["num_layers"],
            dropout=lstm_cfg["dropout"],
            use_layer_norm=lstm_cfg["use_layer_norm"],
        )

        device = self.train_cfg["training"]["device"]
        model_lstm = train_lstm_model(model_lstm, train_loader, val_loader, self.train_cfg, device=device)

        # Evaluate LSTM on test
        test_dataset = SequenceDataset(X_test, y_test, input_length, horizon)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)

        preds_list = []
        y_list = []
        model_lstm.eval()
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                preds = model_lstm(xb).cpu().numpy()
                preds_list.append(preds)
                y_list.append(yb.numpy())
        preds_test = torch.tensor(np.concatenate(preds_list, axis=0)).numpy()
        y_test_seq = np.concatenate(y_list, axis=0)

        logger.info(f"Enhanced LSTM Test MAE: {mae(y_test_seq, preds_test):.3f}")
        logger.info(f"Enhanced LSTM Test RMSE: {rmse(y_test_seq, preds_test):.3f}")
        logger.info(f"Enhanced LSTM Test MAPE: {mape(y_test_seq, preds_test):.3f}")

        # Hybrid model
        models_hybrid, ensemble = train_hybrid_model(
            df_feat, self.data_cfg, self.model_cfg, self.train_cfg, device=device
        )
        logger.info("Hybrid Transformer–ResLSTM + VMD training completed.")
