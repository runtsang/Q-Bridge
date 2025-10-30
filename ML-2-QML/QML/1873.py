"""GraphQNN__gen058: Quantum graph neural network utilities with Pennylane.

The module keeps the original public API (feedforward, fidelity_adjacency,
random_network, random_training_data, state_fidelity) but adds a
variational circuit that can be trained with PennyLane.
"""

from __future__ import annotations

import itertools
import numpy as np
import networkx as nx
import pennylane as qml
import scipy.linalg as la
from typing import Iterable, Sequence, Tuple, List

Tensor = np.ndarray


def _random_qubit_unitary(num_qubits: int) -> Tensor:
    """Return a random unitary matrix of dimension 2**num_qubits."""
    dim = 2 ** num_qubits
    mat = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    q, _ = la.qr(mat)
    return q


def _random_qubit_state(num_qubits: int) -> Tensor:
    """Return a random pure state vector."""
    dim = 2 ** num_qubits
    vec = np.random.randn(dim) + 1j * np.random.randn(dim)
    vec /= la.norm(vec)
    return vec


def random_training_data(target_unitary: Tensor, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate training pairs (state, target_state)."""
    data: List[Tuple[Tensor, Tensor]] = []
    for _ in range(samples):
        state = _random_qubit_state(target_unitary.shape[0].bit_length() - 1)
        data.append((state, target_unitary @ state))
    return data


def random_network(qnn_arch: List[int], samples: int):
    """Build a random network of unitary layers."""
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: List[List[Tensor]] = [[]]
    for layer in range(1, len(qnn_arch)):
        in_qubits = qnn_arch[layer - 1]
        out_qubits = qnn_arch[layer]
        layer_ops: List[Tensor] = []
        for _ in range(out_qubits):
            op = _random_qubit_unitary(in_qubits + 1)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training_data, target_unitary


def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[Tensor]],
    samples: Iterable[Tuple[Tensor, Tensor]],
) -> List[List[Tensor]]:
    """Propagate each sample through the network and record states."""
    stored: List[List[Tensor]] = []
    for state, _ in samples:
        layerwise = [state]
        current = state
        for layer, ops in enumerate(unitaries[1:], start=1):
            # combine all ops in this layer into a single unitary
            combined = ops[0]
            for op in ops[1:]:
                combined = op @ combined
            current = combined @ current
            layerwise.append(current)
        stored.append(layerwise)
    return stored


def state_fidelity(a: Tensor, b: Tensor) -> float:
    """Squared absolute overlap of two pure states."""
    return abs(np.vdot(a, b)) ** 2


def fidelity_adjacency(
    states: Sequence[Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a weighted graph from pairwise fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# ----------------------------------------------------------------------
# Variational circuit and training utilities
# ----------------------------------------------------------------------
class VariationalQNN:
    """A simple parameterised QNN built with PennyLane.

    The circuit consists of a chain of RY rotations on each qubit that
    can be trained to approximate a target unitary.
    """

    def __init__(self, arch: Sequence[int]):
        self.arch = arch
        self.num_qubits = arch[-1]
        self.device = qml.device("default.qubit", wires=self.num_qubits)
        # one rotation per output qubit per layer
        self.num_params = sum(arch[1:])
        self.params = np.random.randn(self.num_params)

        @qml.qnode(self.device)
        def circuit(params: np.ndarray) -> Tensor:
            # initialise |0>
            for w in range(self.num_qubits):
                qml.Hadamard(wires=w)  # simple entangling initialisation
            idx = 0
            for out_qubits in arch[1:]:
                for _ in range(out_qubits):
                    qml.RY(params[idx], wires=idx)
                    idx += 1
            return qml.state()

        self.circuit = circuit

    def forward(self, params: np.ndarray | None = None) -> Tensor:
        return self.circuit(params if params is not None else self.params)


def train_variational(
    vqnn: VariationalQNN,
    target_unitary: Tensor,
    epochs: int = 200,
    lr: float = 0.01,
) -> np.ndarray:
    """Train the variational QNN to approximate the target unitary."""
    opt = qml.GradientDescentOptimizer(stepsize=lr)
    params = vqnn.params.copy()
    for _ in range(epochs):
        params = opt.step(lambda p: loss_fn(p, vqnn, target_unitary), params)
    return params


def loss_fn(params: np.ndarray, vqnn: VariationalQNN, target_unitary: Tensor) -> float:
    """Fidelity‑based loss between the circuit output and target unitary."""
    output = vqnn.circuit(params)
    # target state is target_unitary applied to |0>
    zero_state = np.zeros((2 ** vqnn.num_qubits,), dtype=complex)
    zero_state[0] = 1.0
    target_state = target_unitary @ zero_state
    return 1.0 - state_fidelity(output, target_state)


__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "VariationalQNN",
    "train_variational",
]
