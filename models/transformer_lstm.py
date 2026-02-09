"""
混合Transformer-LSTM模型
结合自注意力机制和时序建模能力
"""
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerLSTM(nn.Module):
    """
    混合Transformer-LSTM模型
    Transformer提取全局依赖，LSTM建模局部时序
    使用残差连接、批量归一化和Dropout防止过拟合
    """
    def __init__(self, input_size: int, d_model: int = 64, nhead: int = 4,
                 num_transformer_layers: int = 2, lstm_hidden_size: int = 128,
                 lstm_num_layers: int = 2, dropout: float = 0.3,
                 fc_hidden_size: int = 64, num_classes: int = 2):
        super().__init__()
        self.model_name = "transformer_lstm"

        # 输入映射
        self.input_proj = nn.Linear(input_size, d_model)
        self.input_ln = nn.LayerNorm(d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.transformer_ln = nn.LayerNorm(d_model)

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=d_model, hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers, batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=False
        )
        self.lstm_ln = nn.LayerNorm(lstm_hidden_size)

        # 残差连接投影 (d_model -> lstm_hidden_size)
        self.residual_proj = nn.Linear(d_model, lstm_hidden_size)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size, fc_hidden_size),
            nn.LayerNorm(fc_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_size, num_classes)
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """Xavier和正交初始化"""
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
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        """x: (batch, seq_len, input_size)"""
        # 输入映射
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        x = self.input_ln(x)

        # Transformer编码
        x = self.pos_encoder(x)
        transformer_out = self.transformer(x)  # (batch, seq_len, d_model)

        # Transformer输出归一化
        transformer_out = self.transformer_ln(transformer_out)

        # LSTM处理
        lstm_out, (h_n, _) = self.lstm(transformer_out)  # lstm_out: (batch, seq_len, hidden)

        # 取最后一个时间步
        lstm_last = lstm_out[:, -1, :]  # (batch, lstm_hidden_size)
        lstm_last = self.lstm_ln(lstm_last)

        # 残差连接: Transformer最后时间步 + LSTM输出
        residual = self.residual_proj(transformer_out[:, -1, :])  # (batch, lstm_hidden_size)
        out = lstm_last + residual

        # 分类
        out = self.classifier(out)
        return out
