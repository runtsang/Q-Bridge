"""Quantum graph neural network using Pennylane variational circuits.

The quantum branch encodes a classical feature vector into a state,
applies a parameterised unitary, and outputs the evolved state.
A fidelity‑based adjacency graph can be constructed from the outputs.
"""

from __future__ import annotations

import itertools
import numpy as np
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import pennylane as qml

__all__ = [
    "GraphQNN",
    "random_network",
    "random_training_data",
    "random_unitary",
]


def random_unitary(dim: int) -> np.ndarray:
    """Generate a random Haar‑distributed unitary of size ``dim``."""
    rng = np.random.default_rng()
    A = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    Q, R = np.linalg.qr(A)
    d = np.diagonal(R)
    ph = d / np.abs(d)
    return Q * ph


def random_training_data(unitary: np.ndarray, samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate synthetic training pairs (input, target)."""
    dataset: List[Tuple[np.ndarray, np.ndarray]] = []
    dim = unitary.shape[0]
    for _ in range(samples):
        ψ = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
        ψ /= np.linalg.norm(ψ)
        target = unitary @ ψ
        dataset.append((ψ, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Return a random target unitary and training data."""
    target = random_unitary(2 ** qnn_arch[-1])
    training_data = random_training_data(target, samples)
    return list(qnn_arch), training_data, target


class GraphQNN:
    """Quantum graph neural network with a Pennylane variational circuit."""

    def __init__(self, arch: Sequence[int], dev: str = "default.qubit", shots: int = 1024):
        self.arch = list(arch)
        self.wires = 2 ** arch[-1]
        self.dev = qml.device(dev, wires=self.wires, shots=shots)
        self.params = np.random.uniform(0, 2 * np.pi, size=self.wires)
        self._circuit = self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.dev)
        def circuit(inputs: np.ndarray, params: np.ndarray):
            # amplitude encoding of the input state
            qml.QubitStateVector(inputs, wires=range(self.wires))
            # parameterised Ry rotations
            for i, θ in enumerate(params):
                qml.RY(θ, wires=i)
            return qml.state()
        return circuit

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Return the quantum state produced by the circuit."""
        return self._circuit(inputs, self.params)

    def train(
        self,
        dataset: List[Tuple[np.ndarray, np.ndarray]],
        lr: float = 0.01,
        epochs: int = 10,
        verbose: bool = False,
    ) -> List[float]:
        """Train the variational parameters using mean‑squared error."""
        opt = qml.AdamOptimizer(stepsize=lr)
        losses: List[float] = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for inp, tgt in dataset:
                def cost(p):
                    pred = self._circuit(inp, p)
                    return np.mean(np.abs(pred - tgt) ** 2)
                self.params = opt.step(cost, self.params)
                epoch_loss += cost(self.params)
            epoch_loss /= len(dataset)
            losses.append(epoch_loss)
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} loss: {epoch_loss:.6f}")
        return losses

    @staticmethod
    def fidelity(a: np.ndarray, b: np.ndarray) -> float:
        """Return the absolute squared overlap between two complex vectors."""
        return np.abs(np.vdot(a, b)) ** 2

    def fidelity_adjacency(
        self,
        states: Sequence[np.ndarray],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Create a weighted graph from state fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
            fid = self.fidelity(a, b)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph
