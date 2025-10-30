"""Hybrid graph‑LSTM module that uses classical graph propagation and a classical LSTM cell.

The module is fully classical and can run on CPU or GPU. It merges the
graph‑based state‑propagation of GraphQNN with a classical LSTM that
processes the node embeddings over time.  An optional quantum LSTM
cell can be passed as a callback, enabling a hybrid training loop.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
# 1. Classical GraphQNN utilities
# --------------------------------------------------------------------------- #

def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix of shape (out, in)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)

def random_training_data(
    weight: Tensor, samples: int
) -> list[Tuple[Tensor, Tensor]]:
    """Generate a dataset of random inputs and the linear target."""
    dataset: list[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset

def random_network(
    qnn_arch: Sequence[int], samples: int
) -> Tuple[list[int], list[Tensor], list[Tuple[Tensor, Tensor]], Tensor]:
    """Build a random graph‑based network and its training data."""
    weights: list[Tensor] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        weights.append(_random_linear(in_f, out_f))
    target_weight = weights[-1]
    training_data = random_training_data(target_weight, samples)
    return list(qnn_arch), weights, training_data, target_weight

def feedforward(
    qnn_arch: Sequence[int],
    weights: Sequence[Tensor],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> list[list[Tensor]]:
    """Propagate a batch of samples through the graph‑based network."""
    stored: list[list[Tensor]] = []
    for features, _ in samples:
        activations = [features]
        current = features
        for weight in weights:
            current = torch.tanh(weight @ current)
            activations.append(current)
        stored.append(activations)
    return stored

def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Return the squared overlap between two unit‑norm tensors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)

def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a weighted graph from state fidelities."""
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
# 2. Classical LSTM cell
# --------------------------------------------------------------------------- #

class ClassicalLSTM(nn.Module):
    """A vanilla LSTM cell implemented with linear gates."""
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

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
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

# --------------------------------------------------------------------------- #
# 3. Hybrid Graph‑LSTM module
# --------------------------------------------------------------------------- #

class GraphQLSTM(nn.Module):
    """Hybrid graph‑LSTM that propagates node features through a graph
    and then processes the resulting embeddings with an LSTM cell.

    Parameters
    ----------
    qnn_arch : Sequence[int]
        Architecture of the graph‑based network (layer sizes).
    hidden_dim : int
        Hidden dimension of the LSTM cell.
    lstm_cell : nn.Module | None, optional
        Custom LSTM cell to use.  If ``None`` a :class:`ClassicalLSTM`
        is instantiated automatically.
    """
    def __init__(
        self,
        qnn_arch: Sequence[int],
        hidden_dim: int,
        lstm_cell: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.qnn_arch = list(qnn_arch)
        self.hidden_dim = hidden_dim
        # Build graph weights
        self.weights = [
            _random_linear(in_f, out_f)
            for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:])
        ]
        # LSTM cell
        self.lstm_cell = (
            lstm_cell
            if lstm_cell is not None
            else ClassicalLSTM(qnn_arch[-1], hidden_dim)
        )

    def forward(
        self,
        graph: nx.Graph,
        node_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        graph : nx.Graph
            The graph structure (currently unused but kept for API compatibility).
        node_features : torch.Tensor
            Tensor of shape (num_nodes, feature_dim) containing
            the initial node embeddings.

        Returns
        -------
        torch.Tensor
            Updated node embeddings of shape (num_nodes, hidden_dim).
        """
        # 1. Propagate through the graph‑based network
        activations = feedforward(
            self.qnn_arch, self.weights, [(node_features, None)]
        )
        # The last activation is the node embeddings after the final layer
        node_embeds = activations[-1][0]  # shape (num_nodes, out_features)

        # 2. Treat each node as a time step and run the LSTM cell
        # Reshape to (seq_len=1, batch=num_nodes, input_dim=out_features)
        seq = node_embeds.unsqueeze(0)
        lstm_out, _ = self.lstm_cell(seq)
        # lstm_out shape: (1, num_nodes, hidden_dim)
        new_node_features = lstm_out[0]  # shape (num_nodes, hidden_dim)

        return new_node_features

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "ClassicalLSTM",
    "GraphQLSTM",
]
