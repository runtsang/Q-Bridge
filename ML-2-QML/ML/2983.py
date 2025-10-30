"""Hybrid classical Self‑Attention + LSTM module for PyTorch."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class ClassicalSelfAttention(nn.Module):
    """Compute self‑attention weighted sum of embeddings."""
    def __init__(self, embed_dim: int, num_heads: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor, rot_params: np.ndarray = None,
                ent_params: np.ndarray = None) -> torch.Tensor:
        # x: (seq_len, batch, embed_dim)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, v)

class ClassicalQLSTM(nn.Module):
    """Classical LSTM cell with linear gates (drop‑in replacement)."""
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forget = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(combined))
            i = torch.sigmoid(self.input(combined))
            g = torch.tanh(self.update(combined))
            o = torch.sigmoid(self.output(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        out = torch.cat(outputs, dim=0)
        return out, (hx, cx)

    def _init_states(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] = None):
        if states is not None:
            return states
        batch = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch, self.hidden_dim, device=device),
                torch.zeros(batch, self.hidden_dim, device=device))

class SelfAttentionQLSTM(nn.Module):
    """Hybrid model: classical self‑attention followed by an LSTM (classical or quantum)."""
    def __init__(self, embed_dim: int, hidden_dim: int, vocab_size: int,
                 tagset_size: int, n_qubits: int = 0, num_heads: int = 1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = ClassicalSelfAttention(embed_dim, num_heads)
        if n_qubits > 0:
            self.lstm = ClassicalQLSTM(embed_dim, hidden_dim)
        else:
            self.lstm = nn.LSTM(embed_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(sentence)            # (seq_len, batch, embed_dim)
        context = self.attention(embeds)             # (seq_len, batch, embed_dim)
        lstm_out, _ = self.lstm(context.unsqueeze(1))  # LSTM expects (seq_len, batch, input)
        logits = self.hidden2tag(lstm_out.squeeze(1))
        return F.log_softmax(logits, dim=1)

__all__ = ["SelfAttentionQLSTM", "ClassicalSelfAttention", "ClassicalQLSTM"]
