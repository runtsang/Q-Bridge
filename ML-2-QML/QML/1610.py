"""GraphQNN__gen176: quantum‑based graph‑neural‑network helper with variational circuits."""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import pennylane as qml

Tensor = np.ndarray
State = List[Tuple[Tensor, Tensor]]


def _random_unitary(num_qubits: int) -> np.ndarray:
    """Generate a random Haar‑distributed unitary over 2**num_qubits."""
    dim = 2 ** num_qubits
    random_matrix = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    q, _ = np.linalg.qr(random_matrix)
    return q


def _random_state(num_qubits: int) -> np.ndarray:
    """Generate a random pure state of num_qubits."""
    dim = 2 ** num_qubits
    vec = np.random.randn(dim) + 1j * np.random.randn(dim)
    vec /= np.linalg.norm(vec)
    return vec


def random_training_data(unitary: np.ndarray, samples: int) -> List[Tuple[Tensor, Tensor]]:
    """Generate input‑output pairs (state, U|state>) for training."""
    dataset = []
    for _ in range(samples):
        state = _random_state(num_qubits=int(np.log2(unitary.shape[0])))
        output = unitary @ state
        dataset.append((state, output))
    return dataset


def random_network(qnn_arch: List[int], samples: int):
    """Create a list of random unitary layers and training data."""
    target_unitary = _random_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    layers: List[List[np.ndarray]] = [[]]
    for layer_idx in range(1, len(qnn_arch)):
        in_q = qnn_arch[layer_idx - 1]
        out_q = qnn_arch[layer_idx]
        layer_ops: List[np.ndarray] = []
        for _ in range(out_q):
            op = _random_unitary(in_q + 1)
            layer_ops.append(op)
        layers.append(layer_ops)

    return qnn_arch, layers, training_data, target_unitary


class QuantumGraphQNN__gen176:
    """
    Quantum variant of GraphQNN that encodes a feed‑forward circuit
    using Pennylane QNodes.  Each layer implements a unitary that
    operates on the previous output plus an ancilla qubit.
    """

    def __init__(self, qnn_arch: Sequence[int], dev_name: str = "default.qubit"):
        self.arch = list(qnn_arch)
        self.dev = qml.device(dev_name, wires=self.arch[-1])
        self.layers = []

    def _build_ansatz(self, inputs: np.ndarray):
        @qml.qnode(self.dev, interface="autograd")
        def circuit():
            qml.QubitStateVector(inputs, wires=range(self.arch[0]))
            # Layerwise unitaries
            for layer_idx, ops in enumerate(self.layers):
                in_q = self.arch[layer_idx]
                for out_idx, unitary in enumerate(ops):
                    ancilla_wire = in_q + out_idx
                    qml.QubitUnitary(unitary, wires=[ancilla_wire] + list(range(in_q)))
            return qml.state()
        return circuit

    def random_initialize(self, qnn_arch: Sequence[int]):
        _, layers, _, _ = random_network(qnn_arch, samples=1)
        self.layers = layers

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Apply the circuit to the input state and return the final state."""
        circuit = self._build_ansatz(inputs)
        return circuit()

    def train_batch(
        self,
        training_data: List[Tuple[Tensor, Tensor]],
        epochs: int = 10,
        lr: float = 0.01,
    ):
        """Placeholder for gradient‑based training of the unitary parameters."""
        pass

    @staticmethod
    def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
        """Overlap squared between two pure states."""
        return abs(a.conj().T @ b) ** 2

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[np.ndarray],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
            fid = QuantumGraphQNN__gen176.state_fidelity(s_i, s_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph


class HybridInferenceQuantum:
    """
    Wrapper that can run either the classical or quantum GraphQNN__gen176
    depending on the flag.
    """

    def __init__(self, classical_qnn: "GraphQNN__gen176", quantum_qnn: QuantumGraphQNN__gen176 | None = None):
        self.classical = classical_qnn
        self.quantum = quantum_qnn

    def run(self, inputs: np.ndarray, use_quantum: bool = False) -> np.ndarray:
        if use_quantum and self.quantum is not None:
            return self.quantum.forward(inputs)
        return self.classical.forward(inputs)


__all__ = [
    "QuantumGraphQNN__gen176",
    "HybridInferenceQuantum",
]
