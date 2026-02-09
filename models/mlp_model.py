"""
MLP模型
"""
import torch
import torch.nn as nn


class MLPModel(nn.Module):
    """多层感知机模型，将序列展平后分类"""

    def __init__(self, input_size: int, seq_length: int = 30,
                 hidden_sizes: list = None, dropout: float = 0.3,
                 num_classes: int = 2):
        super().__init__()
        self.model_name = "mlp"
        if hidden_sizes is None:
            hidden_sizes = [256, 128, 64]

        flat_size = input_size * seq_length
        layers = []
        in_size = flat_size
        for hs in hidden_sizes:
            layers.extend([
                nn.Linear(in_size, hs),
                nn.LayerNorm(hs),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_size = hs

        layers.append(nn.Linear(in_size, num_classes))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        # x: (batch, seq_len, features) -> flatten
        x = x.reshape(x.size(0), -1)
        return self.network(x)
