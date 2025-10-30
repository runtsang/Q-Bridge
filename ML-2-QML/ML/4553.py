"""Hybrid self‑attention LSTM classifier with classical components.

The module combines:
* Classical self‑attention (query/key/value projections)
* Classical LSTM cell (drop‑in replacement of quantum gates)
* Fully‑connected layer (tanh‑activated linear)
* Hybrid dense head (sigmoid with optional shift)

The class exposes a single forward method that accepts a token sequence and returns
a probability distribution over two classes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Classical self‑attention
# --------------------------------------------------------------------------- #
class ClassicalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj   = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)
        scores = torch.softmax(Q @ K.transpose(-2, -1) / (self.embed_dim ** 0.5), dim=-1)
        return scores @ V


# --------------------------------------------------------------------------- #
# Classical LSTM cell (drop‑in replacement)
# --------------------------------------------------------------------------- #
class ClassicalQLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits   = n_qubits

        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear  = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
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

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx


# --------------------------------------------------------------------------- #
# Hybrid dense head (classical surrogate for a quantum expectation)
# --------------------------------------------------------------------------- #
class Hybrid(nn.Module):
    def __init__(self, in_features: int, shift: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift  = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = inputs.view(inputs.size(0), -1)
        return torch.sigmoid(logits + self.shift)


# --------------------------------------------------------------------------- #
# Fully‑connected layer (tanh‑activated linear)
# --------------------------------------------------------------------------- #
class FCL(nn.Module):
    def __init__(self, n_features: int = 1):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        # thetas shape: (batch, hidden_dim)
        values = self.linear(thetas)          # (batch, 1)
        return torch.tanh(values).mean(dim=1, keepdim=True)  # (batch, 1)


# --------------------------------------------------------------------------- #
# Main hybrid classifier
# --------------------------------------------------------------------------- #
class HybridSelfAttentionQLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = ClassicalSelfAttention(embed_dim)
        self.lstm      = ClassicalQLSTM(embed_dim, hidden_dim, n_qubits)
        self.fcl       = FCL()
        self.hybrid    = Hybrid(hidden_dim, shift=0.0)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        # sentence: (seq_len, batch)
        x = self.embedding(sentence)                     # (seq_len, batch, embed_dim)
        attn_out = self.attention(x)                     # (seq_len, batch, embed_dim)
        lstm_out, _ = self.lstm(attn_out)                # (seq_len, batch, hidden_dim)
        lstm_avg = lstm_out.mean(dim=0)                  # (batch, hidden_dim)
        fcl_out  = self.fcl(lstm_avg)                    # (batch, 1)
        logits   = self.hybrid(fcl_out)                   # (batch, 1)
        return torch.cat((logits, 1 - logits), dim=-1)    # (batch, 2)


__all__ = ["HybridSelfAttentionQLSTMClassifier"]
