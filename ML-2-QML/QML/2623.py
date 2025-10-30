"""Quantum‑only regression model using Pennylane.

The module implements a variational circuit that encodes the classical
features into a superposition state and measures the expectation of
Pauli‑Z on each qubit.  The resulting feature vector is passed through a
single linear layer to produce a regression output.  The code also
provides a fidelity‑based graph construction that mirrors the GraphQNN
utility, enabling analysis of the quantum states produced by the circuit.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Tuple

import networkx as nx
import numpy as np
import pennylane as qml
import pennylane.numpy as qnp


def generate_superposition_data(num_features: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate dataset of classical features and labels."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class QuantumRegressionFusion:
    """Quantum‑only regression model based on a Pennylane device.

    Parameters
    ----------
    num_features : int
        Number of classical input features.
    num_wires : int
        Number of qubits used in the variational circuit.
    device : str | qml.Device, optional
        Pennylane device to use.  By default a ``default.qubit`` simulator
        with the specified number of wires is created.
    """

    def __init__(self, num_features: int, num_wires: int, device: str | qml.Device | None = None):
        self.num_features = num_features
        self.num_wires = num_wires
        self.device = device or qml.device("default.qubit", wires=num_wires)

        # Trainable parameters for the variational circuit
        self.params = qnp.random.randn(num_wires, 3)  # RX, RY, RZ per qubit

        # Classical head
        self.head_weight = qnp.random.randn(num_wires, 1)
        self.head_bias = 0.0

        # Quantum node
        self.qnode = qml.QNode(self._circuit, self.device)

    def _circuit(self, x: np.ndarray, params: np.ndarray) -> List[float]:
        """Variational circuit that encodes `x` and applies rotations."""
        qml.templates.BasicEntanglerLayers(params, wires=range(self.num_wires))
        for i, val in enumerate(x):
            # Encode each feature into a rotation on its corresponding qubit
            qml.RX(val, wires=i % self.num_wires)
        return [qml.expval(qml.PauliZ(i)) for i in range(self.num_wires)]

    def __call__(self, x: np.ndarray, target: np.ndarray | None = None) -> np.ndarray:
        """Forward pass.  If `target` is provided, returns the MSE loss."""
        # Compute quantum feature vector
        q_features = self.qnode(x, self.params)

        # Classical head
        pred = qnp.dot(q_features, self.head_weight).squeeze() + self.head_bias

        if target is not None:
            loss = qnp.mean((pred - target) ** 2)
            return loss
        return pred

    def build_graph(
        self,
        states: Iterable[np.ndarray],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Build a graph from the quantum feature vectors."""
        def _fid(a: np.ndarray, b: np.ndarray) -> float:
            a_norm = a / (np.linalg.norm(a) + 1e-12)
            b_norm = b / (np.linalg.norm(b) + 1e-12)
            return float(np.dot(a_norm, b_norm) ** 2)

        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = _fid(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph


__all__ = ["QuantumRegressionFusion", "generate_superposition_data"]
