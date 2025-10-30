"""GraphQNN__gen409: Classical graph neural network with hybrid training and spectral analysis.

The module retains the original public API (feedforward, fidelity_adjacency,
random_network, random_training_data, state_fidelity) but wraps them in a
`GraphQNN` class that exposes a `HybridTrainer` for end‑to‑end optimisation
of a classical GNN and a quantum‑inspired layer.  The fidelity‑based graph
construction can optionally return Laplacian eigen‑vectors for spectral
clustering.

Author: OpenAI GPT‑OSS-20b
"""

from __future__ import annotations

import itertools
import numpy as np
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Optional

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

Tensor = torch.Tensor


# --------------------------------------------------------------------------- #
# 1.  Utility functions  ----------------------------------------------------- #
# --------------------------------------------------------------------------- #
def _random_linear(in_features: int, out_features: int) -> Tensor:
    """Return a random (out, in) weight matrix."""
    return torch.randn(out_features, in_features, dtype=torch.float32)


def random_training_data(weight: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate a dataset of (input, target) pairs for a linear layer."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        features = torch.randn(weight.size(1), dtype=torch.float32)
        target = weight @ features
        dataset.append((features, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random classical network and a training set for the last layer."""
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
    """Forward‑propagate each sample through the linear network."""
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
    """Squared overlap between two unit‑norm vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.dot(a_norm, b_norm).item() ** 2)


def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: Optional[float] = None,
    secondary_weight: float = 0.5,
    spectral: bool = False,
    k: int = 5,
) -> nx.Graph:
    """Build a weighted graph from pairwise fidelities.

    Parameters
    ----------
    states : Sequence[Tensor]
        Pure‑state vectors.
    threshold : float
        Primary fidelity cutoff.
    secondary : Optional[float], optional
        Secondary threshold for a weaker connection.
    secondary_weight : float, optional
        Weight given to secondary edges.
    spectral : bool, optional
        If True, attach the first *k* Laplacian eigen‑vectors to each node.
    k : int, optional
        Number of eigen‑vectors to store when *spectral* is True.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    if spectral:
        lap = nx.laplacian_matrix(graph).astype(float).todense()
        eigvals, eigvecs = np.linalg.eigh(lap)
        for idx in range(len(states)):
            graph.nodes[idx]["eigenvec"] = eigvecs[:, idx][:k].tolist()
    return graph


# --------------------------------------------------------------------------- #
# 2.  GraphQNN class -------------------------------------------------------- #
# --------------------------------------------------------------------------- #
class GraphQNN:
    """Container for classical GNN utilities and a hybrid training helper."""
    def __init__(self, qnn_arch: Sequence[int], device: str = "cpu"):
        self.qnn_arch = list(qnn_arch)
        self.device = device
        self.weights: List[Tensor] = [_random_linear(in_f, out_f).to(device)
                                     for in_f, out_f in zip(self.qnn_arch[:-1], self.qnn_arch[1:])]

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int):
        return random_network(qnn_arch, samples)

    @staticmethod
    def random_training_data(weight: Tensor, samples: int):
        return random_training_data(weight, samples)

    def feedforward(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[List[Tensor]]:
        return feedforward(self.qnn_arch, self.weights, samples)

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        return state_fidelity(a, b)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: Optional[float] = None,
        secondary_weight: float = 0.5,
        spectral: bool = False,
        k: int = 5,
    ) -> nx.Graph:
        return fidelity_adjacency(
            states,
            threshold,
            secondary=secondary,
            secondary_weight=secondary_weight,
            spectral=spectral,
            k=k,
        )


# --------------------------------------------------------------------------- #
# 3.  Hybrid training helper ------------------------------------------------- #
# --------------------------------------------------------------------------- #
class QuantumLayer(nn.Module):
    """Simple variational layer implemented with PennyLane."""
    def __init__(self, num_qubits: int, num_layers: int = 2):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.dev = qml.device("default.qubit", wires=num_qubits)
        self.weights = nn.Parameter(torch.randn(num_layers, num_qubits, 3))
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _circuit(self, inputs: Tensor, weights: Tensor) -> Tensor:
        for i in range(self.num_qubits):
            qml.RX(inputs[i], wires=i)
        for layer in range(self.num_layers):
            for i in range(self.num_qubits):
                qml.Rot(*weights[layer, i], wires=i)
            # simple entangling pattern
            for i in range(self.num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

    def forward(self, inputs: Tensor) -> Tensor:
        return self.qnode(inputs, self.weights)


class HybridTrainer:
    """End‑to‑end optimiser for a classical GNN and a quantum layer."""
    def __init__(
        self,
        gnn: nn.Module,
        qlayer: nn.Module,
        lr: float = 1e-3,
    ):
        self.gnn = gnn
        self.qlayer = qlayer
        self.optimizer = torch.optim.Adam(
            list(gnn.parameters()) + list(qlayer.parameters()), lr=lr
        )

    def train_step(self, x: Tensor, y: Tensor) -> float:
        """Single optimisation step."""
        self.optimizer.zero_grad()
        z = self.gnn(x)
        q_out = self.qlayer(z)
        loss = F.mse_loss(q_out, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()
