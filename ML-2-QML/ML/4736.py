"""Unified classical attention‑LSTM‑graph module.

The module implements:

1. `MultiHeadAttention` – scalable self‑attention (NumPy/PyTorch).
2. `ClassicalQLSTM` – classical LSTM cell (from seed).
3. `GraphHelpers` – state fidelity and adjacency utilities.
4. `UnifiedModel` – thin wrapper that ties everything together.

It deliberately merges ideas from:
- Classical attention from *SelfAttention.py*.
- Classical LSTM from *QLSTM.py*.
- Graph utilities from *GraphQNN.py*.

All components are pure Python/PyTorch and fully importable.
"""

from __future__ import annotations

import itertools
import math
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, List, Sequence, Tuple

__all__ = [
    "MultiHeadAttention",
    "ClassicalQLSTM",
    "GraphHelpers",
    "UnifiedModel",
]

# --------------------------------------------------------------------------- #
# 1. Multi‑Head Attention
# --------------------------------------------------------------------------- #
class MultiHeadAttention(nn.Module):
    """
    Scalable multi‑head self‑attention.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    num_heads : int
        Number of attention heads.
    dropout : float, optional
        Dropout probability on the attention weights.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        hidden_states : torch.Tensor
            Tensor of shape (seq_len, batch, embed_dim).

        Returns
        -------
        torch.Tensor
            Tensor of shape (seq_len, batch, embed_dim) after self‑attention.
        """
        seq_len, batch, _ = hidden_states.size()

        # Linear projections
        q = self.q_proj(hidden_states).view(seq_len, batch, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(seq_len, batch, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(seq_len, batch, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot‑product
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(seq_len, batch, self.embed_dim)
        return self.out_proj(attn_output)

# --------------------------------------------------------------------------- #
# 2. Classical QLSTM (drop‑in replacement)
# --------------------------------------------------------------------------- #
class ClassicalQLSTM(nn.Module):
    """
    Classical LSTM cell (drop‑in replacement for quantum version).
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
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
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

# --------------------------------------------------------------------------- #
# 3. Graph helpers (state fidelity + adjacency)
# --------------------------------------------------------------------------- #
class GraphHelpers:
    """
    Utilities for building fidelity‑based graphs from hidden states.
    """

    @staticmethod
    def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
        """
        Cosine‑like fidelity for real tensors.
        """
        a_norm = a / (torch.norm(a) + 1e-12)
        b_norm = b / (torch.norm(b) + 1e-12)
        return float(torch.dot(a_norm, b_norm).item() ** 2)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[torch.Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """
        Build a weighted graph from pairwise state fidelities.
        """
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphHelpers.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

# --------------------------------------------------------------------------- #
# 4. Unified model
# --------------------------------------------------------------------------- #
class UnifiedModel(nn.Module):
    """
    End‑to‑end model that chains embedding → multi‑head attention → (classical) QLSTM → tagger.

    Parameters
    ----------
    embed_dim : int
        Embedding dimensionality.
    hidden_dim : int
        Hidden size of the LSTM.
    vocab_size : int
        Size of the vocabulary.
    tagset_size : int
        Number of output tags.
    num_heads : int
        Number of attention heads.
    n_qubits : int
        Number of qubits for the quantum LSTM variant (ignored here).
    use_q_lstm : bool
        If True, use the quantum LSTM; otherwise use classical QLSTM.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        num_heads: int = 4,
        n_qubits: int = 0,
        use_q_lstm: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embed_dim)

        self.attention = MultiHeadAttention(embed_dim, num_heads)

        if use_q_lstm:
            # Import here to avoid heavy dependencies when not used
            from.qml_code import QuantumQLSTM  # type: ignore
            self.lstm = QuantumQLSTM(embed_dim, hidden_dim, n_qubits)
        else:
            self.lstm = ClassicalQLSTM(embed_dim, hidden_dim, n_qubits)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        sentence : torch.Tensor
            Tensor of shape (seq_len,) containing token indices.

        Returns
        -------
        torch.Tensor
            Log‑softmax scores for each tag (seq_len, tagset_size).
        """
        embeds = self.word_embeddings(sentence)
        # Reshape for attention: (seq_len, batch=1, embed_dim)
        embeds = embeds.unsqueeze(1)
        attn_out = self.attention(embeds).squeeze(1)  # (seq_len, embed_dim)
        lstm_out, _ = self.lstm(attn_out.unsqueeze(1))  # (seq_len, 1, hidden_dim)
        tag_logits = self.hidden2tag(lstm_out.squeeze(1))  # (seq_len, tagset_size)
        return F.log_softmax(tag_logits, dim=1)

    def hidden_state_graph(
        self,
        hidden_states: torch.Tensor,
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """
        Build a fidelity graph from a sequence of hidden states.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Tensor of shape (seq_len, hidden_dim).
        """
        states = [hidden_states[i] for i in range(hidden_states.size(0))]
        return GraphHelpers.fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)
