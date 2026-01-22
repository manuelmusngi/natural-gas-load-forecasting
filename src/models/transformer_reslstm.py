import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, T, D)
        T = x.size(1)
        return x + self.pe[:, :T, :]

class TransformerResLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        forecast_horizon: int,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        reslstm_hidden_size: int = 128,
        reslstm_num_layers: int = 2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.pos_encoder = PositionalEncoding(d_model)
        self.reslstm = nn.LSTM(
            input_size=d_model,
            hidden_size=reslstm_hidden_size,
            num_layers=reslstm_num_layers,
            batch_first=True,
            dropout=dropout if reslstm_num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(reslstm_hidden_size, forecast_horizon)

    def forward(self, x):
        # x: (B, T, F)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        out, _ = self.reslstm(x)
        out = out[:, -1, :]
        out = self.head(out)
        return out
