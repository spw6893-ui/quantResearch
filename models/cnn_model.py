"""
CNN时序分类模型 (带残差连接)
"""
import torch
import torch.nn as nn


class ResidualConvBlock(nn.Module):
    """带残差连接的Conv1d块"""

    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.3):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.residual_proj = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.residual_proj(x)
        out = self.drop(self.act(self.bn(self.conv(x))))
        return out + residual


class CNNModel(nn.Module):
    """1D CNN用于时序分类 (带残差连接)"""

    def __init__(self, input_size: int, seq_length: int = 30,
                 num_filters: list = None, kernel_sizes: list = None,
                 dropout: float = 0.3, fc_hidden_size: int = 64,
                 **kwargs):
        super().__init__()
        self.model_name = "cnn"
        if num_filters is None:
            num_filters = [32, 64, 128]
        if kernel_sizes is None:
            kernel_sizes = [3, 3, 3]

        blocks = []
        in_channels = input_size
        for nf, ks in zip(num_filters, kernel_sizes):
            blocks.append(ResidualConvBlock(in_channels, nf, ks, dropout))
            in_channels = nf

        self.conv_layers = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 单输出用于BCE
        self.classifier = nn.Sequential(
            nn.Linear(num_filters[-1], fc_hidden_size),
            nn.BatchNorm1d(fc_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_size, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        """x: (batch, seq_len, features) -> (batch,) logits"""
        x = x.transpose(1, 2)
        x = self.conv_layers(x)
        x = self.global_pool(x).squeeze(-1)
        out = self.classifier(x)
        return out.squeeze(-1)
