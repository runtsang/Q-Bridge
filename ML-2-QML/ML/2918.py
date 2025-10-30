"""Hybrid LSTM‑Graph module: classical implementation.

This module defines:

* :class:`UnifiedTagger` – a sequence tagging model that uses a
  standard :class:`torch.nn.LSTM` cell.  It accepts the same
  constructor arguments as the original QLSTMTagger, but the
  ``n_qubits`` flag toggles the use of a quantum‑enhanced
  implementation (which is *not* available in this classical
  module).  The class is fully compatible with the anchor
  ``QLSTM.py`` but does not import any quantum libraries.

* Utility functions that mirror the GraphQNN helpers:
  ``random_network``, ``feedforward``, ``state_fidelity``,
  ``fidelity_adjacency``.  They operate on tensors and
  produce a graph of hidden states that can be used to analyse
  similarity between time‑step representations.

The design keeps the public API unchanged so downstream code can
switch between the classical and quantum versions by simply
importing the appropriate module.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Tuple, Sequence

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
# 1. Classical utilities (graph based on hidden states)
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate training data where target = weight @ feature."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random feed‑forward network and training data."""
    weights: List[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Simple feed‑forward propagation returning all layer activations."""
    stored: List[List[Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored

def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared inner‑product between two normalized vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm @ b_norm).item() ** 2)

def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a weighted adjacency graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# --------------------------------------------------------------------------- #
# 2. Classical LSTM cell (drop‑in replacement)
# --------------------------------------------------------------------------- #
class UnifiedQLSTM(nn.Module):
    """Drop‑in classical LSTM cell mirroring the quantum interface."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.linear_forget = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.linear_input = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.linear_update = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.linear_output = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(
        self,
        inputs: Tensor,
        states: Tuple[Tensor, Tensor] | None = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.linear_forget(combined))
            i = torch.sigmoid(self.linear_input(combined))
            g = torch.tanh(self.linear_update(combined))
            o = torch.sigmoid(self.linear_output(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(
        self,
        inputs: Tensor,
        states: Tuple[Tensor, Tensor] | None,
    ) -> Tuple[Tensor, Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

# --------------------------------------------------------------------------- #
# 3. Tagger model (sequence labeling)
# --------------------------------------------------------------------------- #
class UnifiedTagger(nn.Module):
    """Sequence tagging model that can use the classical LSTM cell."""
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
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = UnifiedQLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: Tensor) -> Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

    # ----------------------------------------------------------------------- #
    # 3a. Graph analysis of hidden states
    # ----------------------------------------------------------------------- #
    def hidden_state_graph(
        self,
        hidden_states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Return a graph built from a list of hidden‑state vectors."""
        return fidelity_adjacency(hidden_states, threshold, secondary=secondary, secondary_weight=secondary_weight)

__all__ = [
    "UnifiedTagger",
    "UnifiedQLSTM",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
