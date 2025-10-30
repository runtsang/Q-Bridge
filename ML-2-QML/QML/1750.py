"""
GraphQNNHybrid.py (quantum)

This module implements a hybrid graph neural network that maps graph node features to qubit registers and applies a parameter‑shared variational quantum circuit using Pennylane.  The network outputs expectation values of Pauli‑Z as new node features.  Utility functions for generating random data and fidelity‑based graphs are also provided.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import pennylane as qml
import torch
from torch_geometric.data import Data

Tensor = torch.Tensor

def _random_unitary(num_qubits: int) -> torch.Tensor:
    """Return a random unitary matrix of size 2^n."""
    dim = 2 ** num_qubits
    mat = torch.randn(dim, dim, dtype=torch.complex64)
    mat, _ = torch.linalg.qr(mat)
    return mat

def random_training_data(unitary: torch.Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate synthetic data for a target unitary."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    dim = unitary.shape[0]
    for _ in range(samples):
        state = torch.randn(dim, dtype=torch.complex64)
        state /= torch.linalg.norm(state)
        target = unitary @ state
        dataset.append((state, target))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random quantum network architecture and associated data."""
    target_unitary = _random_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)
    unitaries: List[List[torch.Tensor]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: List[torch.Tensor] = []
        for _ in range(num_outputs):
            op = _random_unitary(num_inputs + 1)
            layer_ops.append(op)
        unitaries.append(layer_ops)
    return qnn_arch, unitaries, training_data, target_unitary

def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Return the absolute squared overlap between two pure state vectors."""
    return abs(torch.vdot(a, b)).item() ** 2

def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a weighted graph from pairwise state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

class GraphQNNHybrid:
    """Hybrid graph neural network that runs a parameter‑shared variational quantum circuit on each node."""
    def __init__(self, arch: Sequence[int], n_qubits: int = 3):
        self.arch = arch
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)
        # Learnable parameters for the variational ansatz: one per qubit (for RY rotations)
        self.params = torch.randn(n_qubits, requires_grad=True)

    def _build_circuit(self, rx_angles: torch.Tensor, ry_angles: torch.Tensor):
        for i in range(self.n_qubits):
            qml.RX(rx_angles[i], wires=i)
            qml.RY(ry_angles[i], wires=i)
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    @property
    def circuit(self):
        return qml.qnode(self.dev, interface="torch")(self._build_circuit)

    def forward(self, data: Data) -> List[Tensor]:
        """Run the variational circuit on each node and return output features."""
        x = data.x
        outputs: List[Tensor] = []
        for node_vec in x:
            # Use the first n_qubits of the node vector as RX angles
            rx_angles = node_vec[:self.n_qubits]
            ry_angles = self.params
            out = self.circuit(rx_angles, ry_angles)
            outputs.append(torch.tensor(out))
        return outputs
