"""Hybrid LSTM for sequence tagging with optional quantum gate regularisation and graph‑based fidelity attention.

The module implements a classical LSTM backbone that can optionally inject a quantum gate
layer (provided by the QML counterpart).  A graph of hidden‑state fidelities is built
and used as a lightweight attention mask.  The public API mirrors that of the original
`QLSTM.py` seed but adds a `quantum_gate` argument to `HybridTagger.forward`.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import itertools

# --------------------------------------------------------------------------- #
#  Classical LSTM backbone with optional quantum gate injection
# --------------------------------------------------------------------------- #

class ClassicalQLSTM(nn.Module):
    """Pure‑classical LSTM with an optional quantum gate wrapper.

    If `quantum_gate` is supplied during forward, the hidden states are passed
    through the gate before the final linear projection.  The gate must be a
    callable that accepts a tensor of shape (batch, seq_len, hidden_dim) and
    returns a tensor of the same shape.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.n_qubits = n_qubits
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        quantum_gate: Optional[callable] = None,
    ) -> torch.Tensor:
        out, _ = self.lstm(x)
        if self.n_qubits > 0 and quantum_gate is not None:
            out = quantum_gate(out)
        out = self.proj(out)
        return out


# --------------------------------------------------------------------------- #
#  Graph‑based fidelity attention
# --------------------------------------------------------------------------- #

def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Squared overlap of two unit‑norm tensors."""
    return float((a * b).sum().abs().item() ** 2)

def fidelity_adjacency(
    states: torch.Tensor,
    threshold: float,
    *,
    secondary: Optional[float] = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a weighted graph from state fidelities.

    Parameters
    ----------
    states: Tensor of shape (batch, hidden_dim)
        Hidden states to compare.
    threshold: float
        Fidelity threshold for a weight of 1.0.
    secondary: Optional[float]
        Lower threshold for a secondary weight.
    secondary_weight: float
        Weight assigned to edges between `secondary` and `threshold`.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(states.size(0)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(a, b)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
#  Hybrid tagger that can switch between classical and quantum LSTM
# --------------------------------------------------------------------------- #

class HybridTagger(nn.Module):
    """Sequence tagging model that accepts a quantum gate during forward.

    The gate is expected to be a callable that transforms the LSTM output
    tensor.  When `quantum_gate` is ``None`` the model behaves exactly like
    a standard classical LSTM tagger.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = ClassicalQLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(
        self,
        sentence: torch.Tensor,
        quantum_gate: Optional[callable] = None,
    ) -> torch.Tensor:
        embeds = self.emb(sentence)
        lstm_out = self.lstm(embeds, quantum_gate=quantum_gate)
        logits = self.hidden2tag(lstm_out)
        return F.log_softmax(logits, dim=-1)


__all__ = ["ClassicalQLSTM", "fidelity_adjacency", "HybridTagger"]
