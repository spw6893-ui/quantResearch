"""
MLP模型 (带残差连接)
"""
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """带残差连接的全连接块"""

    def __init__(self, in_size, out_size, dropout):
        super().__init__()
        self.fc = nn.Linear(in_size, out_size)
        self.bn = nn.BatchNorm1d(out_size)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.residual_proj = nn.Linear(in_size, out_size) if in_size != out_size else nn.Identity()

    def forward(self, x):
        residual = self.residual_proj(x)
        out = self.drop(self.act(self.bn(self.fc(x))))
        return out + residual


class MLPModel(nn.Module):
    """多层感知机模型，带残差连接"""

    def __init__(self, input_size: int, seq_length: int = 30,
                 hidden_sizes: list = None, dropout: float = 0.3,
                 **kwargs):
        super().__init__()
        self.model_name = "mlp"
        if hidden_sizes is None:
            hidden_sizes = [256, 128, 64]

        flat_size = input_size * seq_length
        blocks = []
        in_size = flat_size
        for hs in hidden_sizes:
            blocks.append(ResidualBlock(in_size, hs, dropout))
            in_size = hs

        self.blocks = nn.Sequential(*blocks)
        # 单输出用于BCE
        self.head = nn.Linear(in_size, 1)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        """x: (batch, seq_len, features) -> (batch,) logits"""
        x = x.reshape(x.size(0), -1)
        x = self.blocks(x)
        out = self.head(x)
        return out.squeeze(-1)
