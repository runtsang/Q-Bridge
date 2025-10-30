"""Hybrid Graph Neural Network module.

This module defines a `HybridGraphQNN` class that combines a
classical graph neural network (GCN) with a variational quantum
circuit.  The quantum circuit is implemented in the accompanying
`qml` module and provides a differentiable expectation value
that is concatenated with the classical embedding before the final
classification layer.

Functions
---------
random_network
random_training_data
state_fidelity
fidelity_adjacency
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import torch_geometric.utils as tg_utils

# Import the quantum layer implementation
from.qml import QuantumLayer

Tensor = torch.Tensor
QObj = object  # placeholder for Qutip objects, not used in this module


def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random weight matrix of shape (out_features, in_features)."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(
    samples: int,
    num_nodes_range: Tuple[int, int] = (5, 20),
    num_node_features: int = 3,
    num_classes: int = 3,
) -> List[Tuple[Data, torch.Tensor]]:
    """Generate a list of random graph samples and integer labels."""
    dataset: List[Tuple[Data, torch.Tensor]] = []
    for _ in range(samples):
        num_nodes = random.randint(num_nodes_range[0], num_nodes_range[1])
        edge_index = tg_utils.erdos_renyi_graph(num_nodes, 0.3, directed=False)
        x = torch.randn(num_nodes, num_node_features, dtype=torch.float32)
        data = Data(x=x, edge_index=edge_index)
        label = torch.tensor([random.randint(0, num_classes - 1)], dtype=torch.long)
        dataset.append((data, label))
    return dataset


def random_network(
    qnn_arch: Sequence[int], samples: int
) -> Tuple[Sequence[int], Tuple[Tensor], List[Tuple[Data, torch.Tensor]], Tuple[Tensor]]:
    """
    Generate a random hybrid architecture.

    Parameters
    ----------
    qnn_arch : Sequence[int]
        Length‑5 sequence: [num_node_features, hidden_dim, num_classes,
        q_layers, q_qubits].
    samples : int
        Number of training samples to generate.

    Returns
    -------
    Tuple containing
        - architecture list,
        - random quantum angles (q_layers, q_qubits),
        - training data,
        - target quantum angles (used as a placeholder for comparison).
    """
    num_node_features, hidden_dim, num_classes, q_layers, q_qubits = qnn_arch
    training_data = random_training_data(samples, num_node_features=num_node_features, num_classes=num_classes)
    quantum_angles = torch.randn((q_layers, q_qubits), dtype=torch.float32)
    target_params = (quantum_angles,)
    return qnn_arch, (quantum_angles,), training_data, target_params


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Return the squared absolute overlap between two expectation vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((torch.dot(a_norm, b_norm).item() ** 2))


def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a weighted adjacency graph from state fidelities.

    Edges with fidelity >= threshold receive weight 1.0; if a secondary
    threshold is provided, fidelities between secondary and threshold
    receive the secondary weight.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


class HybridGraphQNN(nn.Module):
    """
    Hybrid Graph Neural Network that combines a classical GCN with a
    variational quantum circuit.

    Parameters
    ----------
    num_node_features : int
        Number of node features in the input graph.
    hidden_dim : int
        Hidden dimension of the GCN.
    num_classes : int
        Number of output classes.
    q_layers : int
        Number of variational layers in the quantum circuit.
    q_qubits : int
        Number of qubits in the quantum circuit.
    """

    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int,
        num_classes: int,
        q_layers: int,
        q_qubits: int,
    ) -> None:
        super().__init__()
        # Classical GCN backbone
        self.gcn = GCNConv(num_node_features, hidden_dim)
        # Linear mapping from classical embedding to quantum angles
        self.angle_mlp = nn.Linear(hidden_dim, q_layers * q_qubits)
        # Quantum refinement block
        self.quantum = QuantumLayer(q_qubits, q_layers)
        # Final classifier
        self.classifier = nn.Linear(hidden_dim + q_qubits, num_classes)

    def forward(self, data: Data) -> Tensor:
        """
        Forward pass.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Graph data object containing ``x`` and ``edge_index``.

        Returns
        -------
        Tensor
            Logits of shape (batch_size, num_classes).
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # Classical GCN
        x = self.gcn(x, edge_index)
        x = F.relu(x)
        # Global pooling to obtain graph‑level embedding
        x = global_mean_pool(x, batch)  # shape: [batch, hidden_dim]
        # Generate quantum angles from classical embedding
        angles = self.angle_mlp(x)  # shape: [batch, q_layers * q_qubits]
        angles = angles.view(-1, self.quantum.q_layers, self.quantum.num_qubits)
        # Quantum refinement
        q_out = self.quantum(angles)  # shape: [batch, q_qubits]
        # Concatenate classical and quantum representations
        combined = torch.cat([x, q_out], dim=-1)  # shape: [batch, hidden_dim + q_qubits]
        # Final classification
        return self.classifier(combined)


def feedforward(
    qnn_arch: Sequence[int],
    weights: Tuple[Tensor],
    samples: Iterable[Tuple[Data, torch.Tensor]],
) -> List[Tensor]:
    """
    Run a forward pass of the hybrid model on a dataset.

    Parameters
    ----------
    qnn_arch : Sequence[int]
        Architecture parameters.
    weights : Tuple[Tensor]
        Tuple containing the quantum angles.
    samples : Iterable[Tuple[Data, torch.Tensor]]
        Iterable of (graph, label) pairs.

    Returns
    -------
    List[Tensor]
        List of logits for each sample.
    """
    num_node_features, hidden_dim, num_classes, q_layers, q_qubits = qnn_arch
    quantum_angles, = weights
    model = HybridGraphQNN(num_node_features, hidden_dim, num_classes, q_layers, q_qubits)
    # Load quantum angles into the model
    with torch.no_grad():
        model.quantum.angles.copy_(quantum_angles)
    model.eval()
    logits = []
    with torch.no_grad():
        for data, _ in samples:
            logits.append(model(data))
    return logits


__all__ = [
    "HybridGraphQNN",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "fidelity_adjacency",
    "feedforward",
]
