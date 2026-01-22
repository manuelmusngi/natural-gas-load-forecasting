import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SequenceDataset(Dataset):
    def __init__(self, X, y, input_length: int, horizon: int):
        self.X = X
        self.y = y
        self.input_length = input_length
        self.horizon = horizon

    def __len__(self):
        return len(self.X) - self.input_length - self.horizon + 1

    def __getitem__(self, idx):
        x_seq = self.X[idx : idx + self.input_length]
        y_seq = self.y[idx + self.input_length : idx + self.input_length + self.horizon]
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_seq, dtype=torch.float32)

def train_lstm_model(model, train_loader, val_loader, cfg, device="cpu"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"],
                                 weight_decay=cfg["training"]["weight_decay"])
    criterion = torch.nn.MSELoss()
    best_val_loss = float("inf")
    patience = cfg["training"]["early_stopping_patience"]
    patience_counter = 0

    for epoch in range(cfg["training"]["num_epochs"]):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip_val"])
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    return model
