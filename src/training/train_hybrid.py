import numpy as np
import torch
from torch.utils.data import DataLoader
from .train_lstm import SequenceDataset, train_lstm_model
from src.models.vmd_decomposition import vmd_decompose
from src.models.transformer_reslstm import TransformerResLSTM
from src.models.ensemble import LinearEnsemble

def train_hybrid_model(df, cfg_data, cfg_model, cfg_train, device="cpu"):
    target_col = cfg_data["dataset"]["target_column"]
    series = df[target_col]
    vmd_cfg = cfg_model["hybrid_transformer_reslstm"]["vmd"]
    modes_df = vmd_decompose(series, num_modes=vmd_cfg["num_modes"])

    X_all = df.drop(columns=[target_col]).values
    y_all = df[target_col].values

    input_length = cfg_model["hybrid_transformer_reslstm"]["input_length"]
    horizon = cfg_model["hybrid_transformer_reslstm"]["forecast_horizon"]

    datasets = []
    models = []
    preds_val_list = []

    # For simplicity, same features for all modes
    for mode_name in modes_df.columns:
        y_mode = modes_df[mode_name].values
        dataset = SequenceDataset(X_all, y_mode, input_length, horizon)
        datasets.append(dataset)

        loader = DataLoader(dataset, batch_size=cfg_train["training"]["batch_size"], shuffle=True)
        # No explicit val split here; in practice, split indices
        model = TransformerResLSTM(
            input_size=X_all.shape[1],
            forecast_horizon=horizon,
            **{k: v for k, v in cfg_model["hybrid_transformer_reslstm"].items() if k not in ["input_length", "forecast_horizon", "vmd"]}
        )
        model = train_lstm_model(model, loader, loader, cfg_train, device=device)
        models.append(model)

    # Build ensemble on validation-like subset
    # Here we just reuse the last part of the dataset as "val"
    N = len(X_all) - input_length - horizon + 1
    X_seq = []
    y_seq = []
    for i in range(N):
        X_seq.append(X_all[i : i + input_length])
        y_seq.append(y_all[i + input_length : i + input_length + horizon])
    X_seq = torch.tensor(np.stack(X_seq), dtype=torch.float32).to(device)
    y_seq = np.stack(y_seq)

    preds_modes = []
    with torch.no_grad():
        for model in models:
            model.to(device)
            preds = model(X_seq).cpu().numpy()
            preds_modes.append(preds)
    preds_modes = np.stack(preds_modes, axis=1)  # (N, M, H)

    ensemble = LinearEnsemble(num_models=len(models))
    ensemble.fit(preds_modes, y_seq)

    return models, ensemble
