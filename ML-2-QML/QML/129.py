"""GraphQNN__gen142: Quantum graph neural network with a variational ansatz.

This module builds on the original seed by replacing the fixed random
unitary generation with a PennyLane variational circuit that can be
optimized against a target unitary.  All classical utilities (feed‑forward,
fidelity, graph construction) are preserved, enabling direct comparison
between classical and quantum models.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import pennylane as qml

Tensor = np.ndarray
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# 1. Utility functions (adapted from the seed)
# --------------------------------------------------------------------------- #
def _random_qubit_unitary(num_qubits: int) -> qml.QubitUnitary:
    """Generate a random unitary as a PennyLane QubitUnitary object."""
    dim = 2 ** num_qubits
    matrix = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
    unitary = qml.math.orth(matrix)
    return qml.QubitUnitary(unitary, wires=range(num_qubits))

def _random_qubit_state(num_qubits: int) -> Tensor:
    """Create a random pure state vector."""
    dim = 2 ** num_qubits
    vec = np.random.normal(size=(dim, 1)) + 1j * np.random.normal(size=(dim, 1))
    vec /= np.linalg.norm(vec)
    return vec

def random_training_data(num_qubits: int, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate (input_state, target_state) pairs for a given target unitary."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, None))  # target to be filled later
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random target unitary and training data."""
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(qnn_arch[-1], samples)
    # Attach target states
    for idx, (state, _) in enumerate(training_data):
        training_data[idx] = (state, target_unitary.matrix @ state)
    return list(qnn_arch), target_unitary, training_data

def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Return the absolute squared overlap between two pure statevectors."""
    return abs(np.vdot(a, b)) ** 2

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
# 2. Variational training loop
# --------------------------------------------------------------------------- #
class GraphQNN:
    """Quantum graph neural network with a variational ansatz."""
    def __init__(
        self,
        qnn_arch: Sequence[int],
        device: str = "default.qubit",
        shots: int = 1024,
    ):
        self.qnn_arch = list(qnn_arch)
        self.num_qubits = self.qnn_arch[-1]
        self.device = qml.device(device, wires=self.num_qubits, shots=shots)
        self.layers = len(self.qnn_arch) - 1
        self.params_shape = (self.layers, self.num_qubits, 3)
        self.params = np.random.randn(*self.params_shape)
        self.target_unitary: qml.QubitUnitary | None = None
        self.training_data: List[Tuple[Tensor, Tensor]] | None = None
        self.circuit = self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.device)
        def circuit(state):
            qml.QubitStateVector(state, wires=range(self.num_qubits))
            qml.templates.StronglyEntanglingLayers(self.params, wires=range(self.num_qubits))
            return qml.state()
        return circuit

    def build_random(self, samples: int = 1000):
        """Generate a random target unitary and training data."""
        _, self.target_unitary, self.training_data = random_network(self.qnn_arch, samples)

    def forward(self, samples: Iterable[Tuple[Tensor, Tensor]]) -> List[Tensor]:
        """Run the variational circuit on the provided samples."""
        if self.circuit is None:
            raise RuntimeError("Build the circuit first.")
        outputs: List[Tensor] = []
        for state, _ in samples:
            outputs.append(self.circuit(state))
        return outputs

    def train_variational(
        self,
        epochs: int = 20,
        lr: float = 0.01,
        verbose: bool = False,
    ) -> None:
        """Optimize the variational parameters to approximate the target unitary."""
        if self.target_unitary is None or self.training_data is None:
            raise RuntimeError("Call build_random() before training.")
        optimizer = qml.GradientDescentOptimizer(stepsize=lr)
        params = self.params
        for epoch in range(epochs):
            params, loss = optimizer.step_and_cost(lambda p: self._cost(p), params)
            self.params = params
            if verbose and (epoch + 1) % 5 == 0:
                log.info(f"Epoch {epoch+1}/{epochs} loss={loss:.4f}")

    def _cost(self, params: np.ndarray) -> float:
        """Cost based on the average fidelity between the circuit output and target."""
        self.params = params
        fidelities: List[float] = []
        for state, target in self.training_data:
            out_state = self.circuit(state)
            fidelities.append(state_fidelity(out_state, target))
        return 1.0 - np.mean(fidelities)

    def predict(self, state: Tensor) -> Tensor:
        """Return the state produced by the circuit for a single input."""
        if self.circuit is None:
            raise RuntimeError("Build the circuit first.")
        return self.circuit(state)

    def build_fidelity_graph(self, state_list: Sequence[Tensor], threshold: float) -> nx.Graph:
        """Construct a graph from state fidelities."""
        return fidelity_adjacency(state_list, threshold)

__all__ = [
    "GraphQNN",
    "random_network",
    "state_fidelity",
    "fidelity_adjacency",
]
