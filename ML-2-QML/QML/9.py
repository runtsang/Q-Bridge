"""GraphQNN__gen007: Quantum MLP using Pennylane.

Mirrors the classical architecture but replaces linear layers with
parameterized unitaries.  The `fit` method optimizes the circuit parameters
to match the target state produced by a random classical linear map.
"""

import pennylane as qml
import pennylane.numpy as np
import networkx as nx
import itertools
from typing import List, Tuple, Sequence

# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------
def _random_qubit_unitary(num_qubits: int) -> np.ndarray:
    dim = 2 ** num_qubits
    matrix = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
    q, _ = np.linalg.qr(matrix)
    return q


def random_training_data(unitary: np.ndarray, samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate synthetic state pairs (input, target) using a fixed unitary."""
    dataset: List[Tuple[np.ndarray, np.ndarray]] = []
    for _ in range(samples):
        state = np.random.normal(size=(2 ** unitary.shape[0], 1)) + 1j * np.random.normal(size=(2 ** unitary.shape[0], 1))
        state /= np.linalg.norm(state)
        target = unitary @ state
        dataset.append((state, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Return architecture, list of unitary matrices, synthetic training data,
    and the target unitary used for data generation."""
    unitaries: List[np.ndarray] = []
    for in_f, out_f in zip(qnn_arch[:-1], qnn_arch[1:]):
        unitaries.append(_random_qubit_unitary(in_f))
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)
    return list(qnn_arch), unitaries, training_data, target_unitary


def fidelity_adjacency(
    states: Sequence[np.ndarray],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Weighted graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = np.abs(np.vdot(a, b)) ** 2
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# ----------------------------------------------------------------------
# Quantum neural network
# ----------------------------------------------------------------------
class GraphQNNQML:
    """Quantum MLP implemented with Pennylane."""

    def __init__(self, arch: Sequence[int], dev: str = "default.qubit"):
        self.arch = list(arch)
        self.dev = dev
        self.num_qubits = arch[-1]
        self.qnode = self._build_qnode()

    def _build_qnode(self):
        @qml.qnode(qml.device(self.dev, wires=self.num_qubits))
        def circuit(inputs, params):
            # Encode input state via rotation gates
            for i, val in enumerate(inputs):
                qml.RY(val.real, wires=i)  # simple encoding
            # Apply parameterized layers
            idx = 0
            for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
                for i in range(out_f):
                    for j in range(in_f):
                        qml.RX(params[idx], wires=i)
                        idx += 1
                # Entangle layer
                for i in range(out_f - 1):
                    qml.CNOT(wires=[i, i + 1])
            return qml.state()
        return circuit

    def fit(
        self,
        samples: int = 200,
        epochs: int = 200,
        lr: float = 0.01,
        fidelity_thresh: float = 0.95,
    ) -> nx.Graph:
        """Optimize circuit parameters to match target states."""
        arch, _, train_data, target_unitary = random_network(self.arch, samples)

        # Flatten all parameters into a single array
        params = np.random.normal(size=self._param_count())
        opt = qml.AdamOptimizer(stepsize=lr)

        for _ in range(epochs):
            params, _ = opt.step_and_cost(
                lambda p: self._cost(p, train_data, target_unitary), params
            )

        # Evaluate final states
        final_states = [self.qnode(state, params) for state, _ in train_data]
        return fidelity_adjacency(final_states, fidelity_thresh)

    def _cost(self, params, train_data, target_unitary):
        loss = 0.0
        for inp, tgt in train_data:
            out = self.qnode(inp, params)
            loss += np.mean(np.abs(out - tgt) ** 2)
        return loss / len(train_data)

    def _param_count(self):
        count = 0
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            count += in_f * out_f
        return count

    def predict(self, inputs: np.ndarray, params: np.ndarray) -> np.ndarray:
        return self.qnode(inputs, params)
