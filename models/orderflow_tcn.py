"""
Orderflow-TCN 模型

用于承载“盘口/订单流微观结构”特征的轻量时序网络：
  - Dilated Conv1d (TCN) 提取多尺度局部模式
  - 残差连接 + BN + Dropout 防止过拟合
  - 全局池化(平均) + 最后时刻拼接，兼顾稳定与时点信息
"""
import torch
import torch.nn as nn


class DilatedResidualBlock(nn.Module):
    """带膨胀卷积的残差块"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.residual_proj = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)
        out = self.drop(self.act(self.bn(self.conv(x))))
        return out + residual


class OrderflowTCN(nn.Module):
    """Orderflow-TCN: (B, T, F) -> logits(B,)"""

    def __init__(
        self,
        input_size: int,
        seq_length: int = 60,
        hidden_channels: int = 64,
        kernel_size: int = 3,
        dilations: list | None = None,
        dropout: float = 0.3,
        fc_hidden_size: int = 64,
        **kwargs,
    ):
        super().__init__()
        self.model_name = "orderflow_tcn"
        if dilations is None:
            dilations = [1, 2, 4, 8]

        self.input_proj = nn.Conv1d(input_size, hidden_channels, kernel_size=1)
        self.input_bn = nn.BatchNorm1d(hidden_channels)

        blocks = []
        in_ch = hidden_channels
        for d in dilations:
            blocks.append(DilatedResidualBlock(in_ch, hidden_channels, kernel_size, d, dropout))
            in_ch = hidden_channels
        self.tcn = nn.Sequential(*blocks)

        # 池化：平均池化 + 最后时刻拼接
        self.pool_avg = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Sequential(
            nn.Linear(hidden_channels * 2, fc_hidden_size),
            nn.BatchNorm1d(fc_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_size, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) -> (B, F, T)
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = self.input_bn(x)
        x = self.tcn(x)

        avg = self.pool_avg(x).squeeze(-1)         # (B, C)
        last = x[:, :, -1]                         # (B, C)
        feat = torch.cat([avg, last], dim=1)       # (B, 2C)
        out = self.head(feat)
        return out.squeeze(-1)

