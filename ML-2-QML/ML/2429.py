import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalAttention(nn.Module):
    """Classical self‑attention block."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, inputs: torch.Tensor,
                rotation_params: np.ndarray,
                entangle_params: np.ndarray) -> torch.Tensor:
        seq_len, batch, embed_dim = inputs.shape
        # Project queries and keys
        q = torch.matmul(inputs, rotation_params.reshape(embed_dim, -1))
        k = torch.matmul(inputs, entangle_params.reshape(embed_dim, -1))
        v = inputs
        scores = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(embed_dim), dim=-1)
        return torch.matmul(scores, v)

class ClassicalQLSTM(nn.Module):
    """Classical LSTM cell (drop‑in replacement for the quantum version)."""
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, inputs: torch.Tensor,
                states: tuple[torch.Tensor, torch.Tensor] | None = None
                ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(self, inputs: torch.Tensor,
                     states: tuple[torch.Tensor, torch.Tensor] | None = None
                     ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

class UnifiedSelfAttentionLSTM(nn.Module):
    """Combined classical self‑attention + LSTM module."""
    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.attention = ClassicalAttention(embed_dim)
        self.lstm = ClassicalQLSTM(embed_dim, hidden_dim)

    def forward(self, inputs: torch.Tensor,
                rotation_params: np.ndarray,
                entangle_params: np.ndarray,
                states: tuple[torch.Tensor, torch.Tensor] | None = None
                ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attn_out = self.attention(inputs, rotation_params, entangle_params)
        lstm_out, (hx, cx) = self.lstm(attn_out, states)
        return lstm_out, (hx, cx)

__all__ = ["UnifiedSelfAttentionLSTM"]
