"""GraphQNN: Quantum graph neural network with a Pennylane variational circuit.

The class implements a variational circuit whose parameters are trained to
mimic a target unitary.  It retains the original feed‑forward, fidelity
and adjacency helpers while adding a training loop based on Adam optimization.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import numpy as np
import networkx as nx
import pennylane as qml
from pennylane import numpy as pnp

Tensor = np.ndarray


def _random_qubit_unitary(num_qubits: int) -> Tensor:
    """Return a random unitary matrix for the given number of qubits."""
    dim = 2**num_qubits
    matrix = pnp.random.normal(size=(dim, dim)) + 1j * pnp.random.normal(size=(dim, dim))
    q, _ = pnp.linalg.qr(matrix)
    return q


def random_training_data(unitary: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate pairs (psi, U psi) for a fixed unitary."""
    dataset: List[Tuple[Tensor, Tensor]] = []
    dim = unitary.shape[0]
    for _ in range(samples):
        psi = pnp.random.normal(size=(dim,)) + 1j * pnp.random.normal(size=(dim,))
        psi /= pnp.linalg.norm(psi)
        dataset.append((psi, unitary @ psi))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random variational circuit and a training set for its last layer."""
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: List[List[Tensor]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: List[Tensor] = []
        for _ in range(num_outputs):
            op = _random_qubit_unitary(num_inputs + 1)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return list(qnn_arch), unitaries, training_data, target_unitary


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Return the absolute squared overlap between two pure states."""
    return abs(pnp.vdot(a, b))**2


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


class GraphQNN:
    """Quantum graph neural network implemented with Pennylane.

    The class exposes a `train` method that optimises a variational circuit to
    approximate a target unitary.  It retains the original feed‑forward,
    fidelity and adjacency helpers.
    """

    def __init__(self, qnn_arch: Sequence[int], dev_name: str = "default.qubit", shots: int = 1000):
        self.qnn_arch = list(qnn_arch)
        self.device = qml.device(dev_name, wires=max(self.qnn_arch), shots=shots)
        self.params = self._init_params()
        self.circuit = self._build_circuit()

    def _init_params(self) -> List[Tensor]:
        """Initialise rotation parameters for each layer."""
        params: List[Tensor] = []
        for layer in range(1, len(self.qnn_arch)):
            n_qubits = self.qnn_arch[layer - 1]
            # 3 parameters per qubit (Rx, Ry, Rz)
            layer_params = pnp.random.uniform(low=-np.pi, high=np.pi, size=(n_qubits, 3))
            params.append(layer_params)
        return params

    def _build_circuit(self) -> qml.QNode:
        """Return a QNode that applies a parameterised circuit per layer."""

        @qml.qnode(self.device)
        def circuit(state: Tensor, params: List[Tensor]) -> Tensor:
            # initialise state
            qml.QubitStateVector(state, wires=range(len(state)))
            # apply layers
            for layer_idx, layer_params in enumerate(params):
                for qubit, (theta_x, theta_y, theta_z) in enumerate(layer_params):
                    qml.RX(theta_x, wires=qubit)
                    qml.RY(theta_y, wires=qubit)
                    qml.RZ(theta_z, wires=qubit)
                # entangle all qubits with a simple chain of CNOTs
                for q in range(len(layer_params) - 1):
                    qml.CNOT(wires=[q, q + 1])
            # return the final state vector
            return qml.state()
        return circuit

    def train(
        self,
        X: Tensor,
        y: Tensor,
        epochs: int = 100,
        lr: float = 0.01,
    ) -> None:
        """Train the variational circuit to minimise mean‑squared error."""
        opt = qml.optimize.AdamOptimizer(stepsize=lr)
        for _ in range(epochs):
            def loss_fn(params):
                preds = self.circuit(X, params)
                return pnp.mean(pnp.sum((preds - y) ** 2, axis=1))
            self.params = opt.step(loss_fn, self.params)

    def predict(self, X: Tensor) -> Tensor:
        """Return the circuit output for the given input states."""
        return self.circuit(X, self.params)

    def feedforward(
        self,
        samples: Iterable[Tuple[Tensor, Tensor]],
    ) -> List[List[Tensor]]:
        """Return the state after each layer for every sample."""
        states_per_sample: List[List[Tensor]] = []
        for state, _ in samples:
            layer_states = [state]
            current_state = state
            for layer_idx, layer_params in enumerate(self.params):
                current_state = self._apply_layer(current_state, layer_params)
                layer_states.append(current_state)
            states_per_sample.append(layer_states)
        return states_per_sample

    def _apply_layer(self, state: Tensor, layer_params: Tensor) -> Tensor:
        """Apply a single variational layer to a state vector."""
        @qml.qnode(self.device)
        def single_layer(state_in: Tensor, params: Tensor) -> Tensor:
            qml.QubitStateVector(state_in, wires=range(len(state_in)))
            for qubit, (theta_x, theta_y, theta_z) in enumerate(params):
                qml.RX(theta_x, wires=qubit)
                qml.RY(theta_y, wires=qubit)
                qml.RZ(theta_z, wires=qubit)
            for q in range(len(params) - 1):
                qml.CNOT(wires=[q, q + 1])
            return qml.state()
        return single_layer(state, layer_params)

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int):
        """Create a random variational circuit and training data."""
        return random_network(qnn_arch, samples)

    @staticmethod
    def random_training_data(unitary: Tensor, samples: int):
        """Generate random training data for a fixed unitary."""
        return random_training_data(unitary, samples)

    @staticmethod
    def state_fidelity(a: Tensor, b: Tensor) -> float:
        return state_fidelity(a, b)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Tensor],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)


__all__ = [
    "GraphQNN",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "fidelity_adjacency",
]
