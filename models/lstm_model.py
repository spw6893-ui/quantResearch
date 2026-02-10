"""
普通LSTM模型
"""
import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """标准LSTM模型，带残差连接和BatchNorm"""

    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.3,
                 fc_hidden_size: int = 64, **kwargs):
        super().__init__()
        self.model_name = "lstm"

        self.input_bn = nn.BatchNorm1d(input_size)

        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.lstm_bn = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)

        # 残差: 将输入最后时步投影到hidden_size
        self.residual_proj = nn.Linear(input_size, hidden_size)

        # 单输出用于BCE
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, fc_hidden_size),
            nn.BatchNorm1d(fc_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_size, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
            elif 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)

    def forward(self, x):
        """x: (batch, seq_len, input_size) -> (batch,) logits"""
        x = x.transpose(1, 2)
        x = self.input_bn(x)
        x = x.transpose(1, 2)

        lstm_out, _ = self.lstm(x)
        last = lstm_out[:, -1, :]
        last = self.lstm_bn(last)
        last = self.dropout(last)

        # 残差连接: 输入最后时步 -> hidden_size
        residual = self.residual_proj(x[:, -1, :])
        last = last + residual

        out = self.classifier(last)
        return out.squeeze(-1)
