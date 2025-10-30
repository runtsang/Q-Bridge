"""Hybrid Self‑Attention Transformer – classical implementation.

This module merges classical self‑attention, a quantum‑style LSTM,
and a transformer backbone.  All components are pure PyTorch
and can be trained end‑to‑end with standard optimisers.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Classical self‑attention helper
# --------------------------------------------------------------------------- #
class ClassicalSelfAttention(nn.Module):
    """Pure‑classical self‑attention block."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(
        self,
        rotation_params: torch.Tensor,
        entangle_params: torch.Tensor,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute self‑attention output.

        Parameters
        ----------
        rotation_params : torch.Tensor
            Matrix used to build the query vectors.
        entangle_params : torch.Tensor
            Matrix used to build the key vectors.
        inputs : torch.Tensor
            Input sequence of shape (seq_len, embed_dim).

        Returns
        -------
        torch.Tensor
            Attention‑weighted representation of shape (seq_len, embed_dim).
        """
        query = torch.matmul(inputs, rotation_params)
        key = torch.matmul(inputs, entangle_params)
        scores = F.softmax(query @ key.t() / math.sqrt(self.embed_dim), dim=-1)
        return scores @ inputs


# --------------------------------------------------------------------------- #
# Classical LSTM cell (quantum‑style gates but implemented classically)
# --------------------------------------------------------------------------- #
class ClassicalQLSTM(nn.Module):
    """LSTM cell that mimics the quantum‑style gate structure."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Linear projections for the four gates
        self.forget_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch, self.hidden_dim, device=device),
            torch.zeros(batch, self.hidden_dim, device=device),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(combined))
            i = torch.sigmoid(self.input_gate(combined))
            g = torch.tanh(self.update_gate(combined))
            o = torch.sigmoid(self.output_gate(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)


# --------------------------------------------------------------------------- #
# Transformer components (classical)
# --------------------------------------------------------------------------- #
class MultiHeadAttentionBase(nn.Module):
    """Base class for multi‑head attention."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

    def separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        return x.view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)

    def attention(self, query: torch.Tensor, key: torch.Tensor,
                  value: torch.Tensor, mask: Optional[torch.Tensor] = None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        return F.softmax(scores, dim=-1), scores

    def downstream(self, query: torch.Tensor, key: torch.Tensor,
                   value: torch.Tensor, batch_size: int,
                   mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.separate_heads(query)
        k = self.separate_heads(key)
        v = self.separate_heads(value)
        attn, _ = self.attention(q, k, v, mask)
        return attn.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)


class MultiHeadAttentionClassical(MultiHeadAttentionBase):
    """Standard multi‑head attention implemented in PyTorch."""

    def __init__(self, embed_dim: int, num_heads: int,
                 dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.combine_heads = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch = x.size(0)
        k = self.k_linear(x)
        q = self.q_linear(x)
        v = self.v_linear(x)
        out = self.downstream(q, k, v, batch, mask)
        return self.combine_heads(out)


class FeedForwardBase(nn.Module):
    """Base class for feed‑forward networks."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FeedForwardClassical(FeedForwardBase):
    """Two‑layer feed‑forward network."""

    def __init__(self, embed_dim: int, ffn_dim: int,
                 dropout: float = 0.1):
        super().__init__(embed_dim, ffn_dim, dropout)
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlockBase(nn.Module):
    """Base class for transformer blocks."""

    def __init__(self, embed_dim: int, num_heads: int,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TransformerBlockClassical(TransformerBlockBase):
    """Standard transformer block."""

    def __init__(self, embed_dim: int, num_heads: int,
                 ffn_dim: int, dropout: float = 0.1):
        super().__init__(embed_dim, num_heads, dropout)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# --------------------------------------------------------------------------- #
# HybridSelfAttentionTransformer – main class
# --------------------------------------------------------------------------- #
class HybridSelfAttentionTransformer(nn.Module):
    """Hybrid transformer that combines classical self‑attention,
    a quantum‑style LSTM, and a transformer backbone.

    Parameters
    ----------
    vocab_size : int
        Number of tokens in the vocabulary.
    embed_dim : int
        Dimension of token embeddings.
    num_heads : int
        Number of attention heads in each transformer block.
    num_blocks : int
        Number of transformer blocks.
    ffn_dim : int
        Hidden dimension of the feed‑forward network.
    hidden_dim : int
        Hidden dimension of the recurrent unit.
    num_classes : int
        Number of output classes.
    dropout : float, optional
        Drop‑out probability.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.self_attention = ClassicalSelfAttention(embed_dim)

        # Recurrent component
        self.lstm = ClassicalQLSTM(embed_dim, hidden_dim)

        # Transformer backbone
        self.transformer = nn.Sequential(
            *[
                TransformerBlockClassical(
                    embed_dim, num_heads, ffn_dim, dropout
                )
                for _ in range(num_blocks)
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input token indices of shape (seq_len, batch).

        Returns
        -------
        torch.Tensor
            Log‑probabilities for each class.
        """
        tokens = self.token_embedding(x.t())  # (batch, seq_len, embed_dim)
        tokens = self.pos_encoder(tokens)

        # Self‑attention block
        seq_len, batch, embed_dim = tokens.size(1), tokens.size(0), tokens.size(2)
        # Flatten batch dimension for self‑attention
        inputs = tokens.reshape(-1, embed_dim)
        rotation_params = torch.eye(embed_dim, device=inputs.device)
        entangle_params = torch.eye(embed_dim, device=inputs.device)
        attn_out = self.self_attention(rotation_params, entangle_params, inputs)
        attn_out = attn_out.reshape(batch, seq_len, embed_dim)

        # LSTM
        lstm_out, _ = self.lstm(attn_out.permute(1, 0, 2))  # (seq_len, batch, embed_dim)

        # Transformer
        x = lstm_out.permute(1, 0, 2)  # (batch, seq_len, embed_dim)
        x = self.transformer(x)
        x = self.dropout(x.mean(dim=1))
        logits = self.classifier(x)
        return F.log_softmax(logits, dim=1)
