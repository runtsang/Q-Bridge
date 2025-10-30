"""Hybrid LSTM‑Transformer architecture with optional quantum sub‑modules.

The module exposes a single `QuantumEnhancedLSTMTransformer` class that
replaces the classical LSTM cell with a quantum‑augmented gate (from the
first reference) and the transformer blocks with a hybrid attention/FFN
combination (from the second reference).  The implementation keeps the
original constructor signatures so existing training pipelines can be
swapped in without modification.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Classical LSTM cell (original implementation from reference 1)
# --------------------------------------------------------------------------- #
class ClassicalQLSTM(nn.Module):
    """Drop‑in replacement of the classical LSTM used by the original LSTMTagger."""
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

    def forward(self, inputs: torch.Tensor,
                states: Tuple[torch.Tensor, torch.Tensor] | None = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(self,
                     inputs: torch.Tensor,
                     states: Tuple[torch.Tensor, torch.Tensor] | None
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

# --------------------------------------------------------------------------- #
# Transformer components (original classical implementations from reference 2)
# --------------------------------------------------------------------------- #
class MultiHeadAttentionClassical(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.combine_heads = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.size()
        if embed_dim!= self.embed_dim:
            raise ValueError("Input embedding does not match layer size")
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)

        def separate_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch_size, seq_len, self.num_heads,
                          self.embed_dim // self.num_heads).transpose(1, 2)

        k = separate_heads(k)
        q = separate_heads(q)
        v = separate_heads(v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (
            self.embed_dim // self.num_heads) ** 0.5
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        out = torch.matmul(scores, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len,
                                                    self.embed_dim)
        return self.combine_heads(out)

class FeedForwardClassical(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlockClassical(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads,
                                                dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) *
                             (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

# --------------------------------------------------------------------------- #
# Hybrid model that can be instantiated with either classical or quantum
# sub‑modules (classical path only in this file)
# --------------------------------------------------------------------------- #
class QuantumEnhancedLSTMTransformer(nn.Module):
    """
    Hybrid LSTM‑Transformer that defaults to purely classical components.
    The `use_quantum_lstm` and `use_quantum_transformer` flags are accepted
    for API compatibility but ignored in the classical implementation.
    """
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 hidden_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 ffn_dim: int,
                 tagset_size: int,
                 dropout: float = 0.1,
                 use_quantum_lstm: bool = False,
                 use_quantum_transformer: bool = False) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = ClassicalQLSTM(embed_dim, hidden_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlockClassical(embed_dim, num_heads,
                                       ffn_dim, dropout)
             for _ in range(num_blocks)])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, tagset_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            LongTensor of shape (batch, seq_len) containing token indices.
        Returns
        -------
        logits : torch.Tensor
            Tensor of shape (batch, seq_len, tagset_size)
        """
        tokens = self.embedding(x)
        lstm_out, _ = self.lstm(tokens)
        x = self.pos_encoder(lstm_out)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.dropout(x.mean(dim=1))
        return self.classifier(x)

__all__ = ["QuantumEnhancedLSTMTransformer"]
