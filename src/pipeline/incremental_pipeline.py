import pandas as pd
from torch.utils.data import DataLoader
from src.training.train_lstm import SequenceDataset
from src.training.incremental_update import incremental_update
from src.utils.logger import get_logger

logger = get_logger(__name__)

class IncrementalPipeline:
    def __init__(self, model, data_cfg, train_cfg):
        self.model = model
        self.data_cfg = data_cfg
        self.train_cfg = train_cfg

    def update_with_new_data(self, df_new: pd.DataFrame, feature_builder, device="cpu"):
        df_feat = feature_builder(df_new, self.data_cfg)
        target_col = self.data_cfg["dataset"]["target_column"]
        X = df_feat.drop(columns=[target_col]).values
        y = df_feat[target_col].values

        input_length = self.train_cfg["training"]["input_length"]
        horizon = self.train_cfg["training"]["forecast_horizon"]

        dataset = SequenceDataset(X, y, input_length, horizon)
        loader = DataLoader(dataset, batch_size=self.train_cfg["training"]["batch_size"], shuffle=True)

        self.model = incremental_update(self.model, loader, self.train_cfg, device=device)
        logger.info("Incremental update completed.")
        return self.model
