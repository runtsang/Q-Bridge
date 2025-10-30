"""Quantum graph neural network utilities with a variational circuit.

Features
--------
* Variational circuit built with Pennylane that approximates a target unitary.
* `feedforward` returns the state vector at the final layer for each sample.
* `train` runs a gradient‑based optimization using a fidelity loss.
* Fidelity‑based adjacency graph construction is retained.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Callable
import random

import networkx as nx
import pennylane as qml
import pennylane.numpy as np
import torch
import numpy as onp

Tensor = torch.Tensor

def _random_complex_vector(dim: int) -> Tensor:
    """Generate a random complex state vector and normalize."""
    vec = torch.randn(dim, dtype=torch.complex64) + 1j * torch.randn(dim, dtype=torch.complex64)
    return vec / torch.norm(vec)

def random_training_data(unitary: np.ndarray, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate samples by applying a fixed unitary to random states."""
    dim = unitary.shape[0]
    data: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        state = _random_complex_vector(dim)
        target = torch.from_numpy(unitary @ state.numpy()).to(torch.complex64)
        data.append((state, target))
    return data

def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random target unitary and a training set for it."""
    dim = 2 ** qnn_arch[-1]
    target_unitary = qml.utils.random_unitary(dim)
    training_data = random_training_data(target_unitary, samples)
    return list(qnn_arch), target_unitary, training_data

def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Absolute squared overlap between two normalized complex vectors."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float(torch.abs(torch.dot(a_norm.conj(), b_norm))**2)

def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

class GraphQNN:
    """Variational graph‑based quantum neural network."""
    def __init__(self, architecture: Sequence[int], device: str = "default.qubit", shots: int = 1000):
        self.arch = list(architecture)
        self.num_qubits = architecture[-1]
        self.num_layers = len(architecture) - 1
        self.dev = qml.device(device, wires=self.num_qubits, shots=shots)
        # Parameters: rotation angles for each layer, qubit, and axis
        self.params = np.random.randn(self.num_layers, self.num_qubits, 3)
        self.circuit = qml.QNode(self._circuit, self.dev, interface='torch')

    def _circuit(self, inputs: Tensor, params: np.ndarray) -> Tensor:
        """Quantum circuit returning the final state vector."""
        # Encode the classical input via RY rotations
        for i, val in enumerate(inputs):
            qml.RY(val, wires=i)
        # Variational layers
        for layer in range(self.num_layers):
            for i in range(self.num_qubits):
                qml.RZ(params[layer, i, 0], wires=i)
                qml.RX(params[layer, i, 1], wires=i)
                qml.RY(params[layer, i, 2], wires=i)
            for i in range(self.num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        return qml.state()

    def forward(self, inputs: Tensor) -> Tensor:
        """Return the state vector produced by the circuit."""
        return self.circuit(inputs, self.params)

    def predict(self, x: Tensor) -> Tensor:
        """Convenience wrapper for a single sample."""
        return self.forward(x)

    def feedforward(
        self, samples: Iterable[Tuple[Tensor, Tensor]]
    ) -> List[Tensor]:
        """Return the state vectors for each input sample."""
        outputs = []
        for inp, _ in samples:
            outputs.append(self.forward(inp))
        return outputs

    def train(
        self,
        training_data: Iterable[Tuple[Tensor, Tensor]],
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        optimizer: torch.optim.Optimizer,
        epochs: int = 100,
        batch_size: int = 32,
    ) -> None:
        """Gradient‑based training loop using the provided loss function."""
        data = list(training_data)
        for epoch in range(epochs):
            random.seed(epoch)
            random.shuffle(data)
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                xs = torch.stack([x for x, _ in batch])
                ys = torch.stack([y for _, y in batch])
                optimizer.zero_grad()
                preds = torch.stack([self.forward(x) for x in xs])
                loss = loss_fn(preds, ys)
                loss.backward()
                optimizer.step()

__all__ = [
    "GraphQNN",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "fidelity_adjacency",
]
