"""
CNN时序分类模型
"""
import torch
import torch.nn as nn


class CNNModel(nn.Module):
    """1D CNN用于时序分类"""

    def __init__(self, input_size: int, seq_length: int = 30,
                 num_filters: list = None, kernel_sizes: list = None,
                 dropout: float = 0.3, fc_hidden_size: int = 64,
                 num_classes: int = 2):
        super().__init__()
        self.model_name = "cnn"
        if num_filters is None:
            num_filters = [32, 64, 128]
        if kernel_sizes is None:
            kernel_sizes = [3, 3, 3]

        layers = []
        in_channels = input_size
        for i, (nf, ks) in enumerate(zip(num_filters, kernel_sizes)):
            layers.extend([
                nn.Conv1d(in_channels, nf, kernel_size=ks, padding=ks//2),
                nn.BatchNorm1d(nf),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_channels = nf

        self.conv_layers = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(num_filters[-1], fc_hidden_size),
            nn.LayerNorm(fc_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_size, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        x = self.conv_layers(x)
        x = self.global_pool(x).squeeze(-1)  # (batch, num_filters[-1])
        return self.classifier(x)
