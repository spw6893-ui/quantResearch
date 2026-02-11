"""
Temporal Fusion Transformer (TFT) for binary classification.
Simplified version for time-series prediction (no static/known-future split).
All input features are treated as observed past covariates.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GatedLinearUnit(nn.Module):
    def __init__(self, input_dim, output_dim=None, dropout=0.0):
        super().__init__()
        output_dim = output_dim or input_dim
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.dense = nn.Linear(input_dim, output_dim * 2)

    def forward(self, x):
        if self.dropout:
            x = self.dropout(x)
        x = self.dense(x)
        return F.glu(x, dim=-1)


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, context_dim=None):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.context_proj = nn.Linear(context_dim, hidden_dim, bias=False) if context_dim else None
        self.elu = nn.ELU()
        self.linear_2 = nn.Linear(hidden_dim, hidden_dim)
        self.glu = GatedLinearUnit(hidden_dim, output_dim, dropout)
        self.add_norm = nn.LayerNorm(output_dim)
        self.skip_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x, context=None):
        residual = self.skip_proj(x)
        h = self.input_proj(x)
        if context is not None and self.context_proj is not None:
            if context.dim() == 2:
                context = context.unsqueeze(1).expand_as(h)
            h = h + self.context_proj(context)
        h = self.linear_2(self.elu(h))
        h = self.glu(h)
        return self.add_norm(h + residual)


class VariableSelectionNetwork(nn.Module):
    """Selects important variables from input features."""
    def __init__(self, input_dim, n_vars, hidden_dim, dropout, context_dim=None):
        super().__init__()
        self.n_vars = n_vars
        self.hidden_dim = hidden_dim

        # Per-variable GRNs
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(1, hidden_dim, hidden_dim, dropout, context_dim)
            for _ in range(n_vars)
        ])

        # Softmax weights over variables
        self.weight_grn = GatedResidualNetwork(
            n_vars * hidden_dim, hidden_dim, n_vars, dropout, context_dim
        )

    def forward(self, x, context=None):
        # x: (batch, seq_len, n_vars)
        var_outputs = []
        for i in range(self.n_vars):
            var_input = x[:, :, i:i+1]  # (batch, seq, 1)
            var_outputs.append(self.var_grns[i](var_input, context))

        # Stack: (batch, seq, n_vars, hidden)
        stacked = torch.stack(var_outputs, dim=2)

        # Flatten for weight computation
        flat = torch.cat(var_outputs, dim=-1)  # (batch, seq, n_vars * hidden)
        weights = F.softmax(self.weight_grn(flat, context), dim=-1)  # (batch, seq, n_vars)
        weights = weights.unsqueeze(-1)  # (batch, seq, n_vars, 1)

        # Weighted sum
        selected = (stacked * weights).sum(dim=2)  # (batch, seq, hidden)
        return selected, weights.squeeze(-1)


class InterpretableMultiHeadAttention(nn.Module):
    """TFT-style attention with shared values across heads for interpretability."""
    def __init__(self, n_head, d_model, dropout=0.0):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.head_dim = d_model // n_head
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, self.head_dim)  # Shared across heads
        self.out_proj = nn.Linear(self.head_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B, T, _ = q.size()
        K_len = k.size(1)

        q_h = self.q_proj(q).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k_h = self.k_proj(k).view(B, K_len, self.n_head, self.head_dim).transpose(1, 2)
        v_h = self.v_proj(v)  # (B, K_len, head_dim) - shared

        attn = torch.matmul(q_h, k_h.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn_probs = self.dropout(F.softmax(attn, dim=-1))

        # Average attention across heads, apply to shared V
        avg_attn = attn_probs.mean(dim=1)  # (B, T, K_len)
        out = torch.matmul(avg_attn, v_h)  # (B, T, head_dim)
        return self.out_proj(out), attn_probs


class TemporalFusionTransformer(nn.Module):
    """
    Simplified TFT for binary classification on time-series.

    Architecture:
    1. Variable Selection Network -> select important features
    2. LSTM encoder -> capture temporal patterns
    3. Interpretable Multi-Head Attention -> long-range dependencies
    4. GRN + classification head -> binary output

    Input: (batch, seq_len, n_features)
    Output: (batch,) logits
    """
    def __init__(self, input_size, d_model=64, nhead=4,
                 lstm_hidden_size=128, lstm_num_layers=2,
                 dropout=0.3, fc_hidden_size=64, **kwargs):
        super().__init__()
        self.model_name = "tft"
        self.d_model = d_model
        self.input_size = input_size

        # 1. Variable Selection
        self.vsn = VariableSelectionNetwork(
            input_dim=input_size, n_vars=input_size,
            hidden_dim=d_model, dropout=dropout
        )

        # 2. LSTM encoder
        self.lstm = nn.LSTM(
            input_size=d_model, hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers, batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0
        )
        self.lstm_proj = nn.Linear(lstm_hidden_size, d_model)
        self.lstm_norm = nn.LayerNorm(d_model)

        # 3. Post-LSTM GRN (gate + skip)
        self.post_lstm_grn = GatedResidualNetwork(d_model, d_model, d_model, dropout)

        # 4. Interpretable Multi-Head Attention
        self.attention = InterpretableMultiHeadAttention(nhead, d_model, dropout)
        self.attn_grn = GatedResidualNetwork(d_model, d_model, d_model, dropout)
        self.attn_norm = nn.LayerNorm(d_model)

        # 5. Final GRN
        self.final_grn = GatedResidualNetwork(d_model, d_model, d_model, dropout)
        self.final_norm = nn.LayerNorm(d_model)

        # 6. Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, fc_hidden_size),
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
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        """x: (batch, seq_len, input_size) -> (batch,) logits"""
        # Variable selection
        selected, var_weights = self.vsn(x)  # (B, T, d_model)

        # LSTM encoding
        lstm_out, _ = self.lstm(selected)  # (B, T, lstm_hidden)
        lstm_out = self.lstm_proj(lstm_out)  # (B, T, d_model)
        lstm_out = self.lstm_norm(lstm_out + selected)  # residual

        # Post-LSTM gating
        gated = self.post_lstm_grn(lstm_out)

        # Causal mask for self-attention
        T = gated.size(1)
        mask = torch.tril(torch.ones(T, T, device=gated.device)).unsqueeze(0).unsqueeze(0)

        # Self-attention
        attn_out, attn_weights = self.attention(gated, gated, gated, mask)
        attn_out = self.attn_grn(attn_out)
        attn_out = self.attn_norm(attn_out + gated)  # residual

        # Final processing
        final = self.final_grn(attn_out)
        final = self.final_norm(final + attn_out)  # residual

        # Take last timestep for classification
        out = final[:, -1, :]  # (B, d_model)
        return self.classifier(out).squeeze(-1)

    def get_variable_importance(self, x):
        """Return variable selection weights for interpretability."""
        with torch.no_grad():
            _, var_weights = self.vsn(x)
        # Average over batch and time
        return var_weights.mean(dim=(0, 1))  # (n_vars,)
