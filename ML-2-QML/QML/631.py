"""Quantum Graph Neural Network implementation using Pennylane.

Features:
- Parameterized variational circuit defined by a layer architecture.
- Classical embeddings are encoded as rotation gates.
- Fidelity based loss and optional hybrid MSE loss.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import pennylane as qml
import pennylane.numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor

def random_training_data(unitary: np.ndarray, samples: int):
    """Generate random states and their transformed versions."""
    dataset = []
    dim = unitary.shape[0]
    for _ in range(samples):
        state = np.random.randn(dim, 1) + 1j * np.random.randn(dim, 1)
        state = state / np.linalg.norm(state)
        target = unitary @ state
        dataset.append((state, target))
    return dataset

def random_network(
    qnn_arch: List[int],
    samples: int,
) -> tuple[list[int], None, List[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Create a synthetic random target unitary and training data."""
    target_unitary = qml.utils.random_unitary(2 ** qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)
    return qnn_arch, None, training_data, target_unitary

def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """Return the absolute squared overlap between pure states."""
    return float(np.abs(np.vdot(a, b)) ** 2)

def fidelity_adjacency(
    states: Sequence[np.ndarray],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a weighted adjacency graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

class GraphQNN__gen282(nn.Module):
    """
    Variational Quantum Graph Neural Network using Pennylane.
    """

    def __init__(
        self,
        arch: Sequence[int],
        device_name: str = "default.qubit",
        use_hybrid: bool = False,
    ):
        super().__init__()
        self.arch = list(arch)
        self.use_hybrid = use_hybrid
        self.device = qml.device(device_name, wires=self.arch[-1])
        # Number of parameters: 3 rotations per qubit per layer + entanglement
        self.num_params = sum(3 * n + (n - 1) for n in self.arch[1:])
        self.params = nn.Parameter(torch.randn(self.num_params))
        self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.device, interface="torch")
        def circuit(x, params):
            # Encode classical data via RY rotations
            for i, val in enumerate(x):
                qml.RY(val, wires=i)
            # Variational layers
            idx = 0
            for n in self.arch[1:]:
                for i in range(n):
                    qml.RX(params[idx], wires=i)
                    idx += 1
                    qml.RY(params[idx], wires=i)
                    idx += 1
                    qml.RZ(params[idx], wires=i)
                    idx += 1
                # Entanglement
                for i in range(n - 1):
                    qml.CNOT(wires=[i, i + 1])
            return qml.state()
        self.circuit = circuit

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: return quantum state vectors for each input."""
        batch = x.shape[0]
        outputs = []
        for i in range(batch):
            state = self.circuit(x[i], self.params)
            outputs.append(state)
        return torch.stack(outputs)

    def train_step(
        self,
        data_loader,
        optimizer,
        criterion,
    ) -> float:
        """Single epoch training step with optional hybrid fidelity loss."""
        self.train()
        total_loss = 0.0
        for batch in data_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            outputs = self.forward(inputs)
            loss = criterion(outputs, targets)
            if self.use_hybrid:
                fid_loss = 0.0
                for out, tgt in zip(outputs, targets):
                    fid = state_fidelity(out.detach().numpy(), tgt.detach().numpy())
                    fid_loss += (1 - fid)
                loss += 0.1 * fid_loss / len(outputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(data_loader)

__all__ = [
    "GraphQNN__gen282",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
