import torch
import torch.nn as nn

class ResidualLSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, use_layer_norm):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.use_layer_norm = use_layer_norm
        self.layer_norm = nn.LayerNorm(hidden_size) if use_layer_norm else None
        self.proj = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        # x: (B, T, F)
        residual = self.proj(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last time step
        if self.use_layer_norm:
            out = self.layer_norm(out)
        out = out + residual[:, -1, :]
        return out

class EnhancedLSTMForecaster(nn.Module):
    def __init__(
        self,
        input_size: int,
        forecast_horizon: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.block = ResidualLSTMBlock(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )
        self.head = nn.Linear(hidden_size, forecast_horizon)

    def forward(self, x):
        # x: (B, T, F)
        features = self.block(x)
        out = self.head(features)
        return out  # (B, H)
