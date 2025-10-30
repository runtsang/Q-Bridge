"""Hybrid quantum‑classical kernel with graph‑based state analysis.

The module defines UnifiedQuantumKernelGraph that uses a Pennylane
variational circuit to encode classical data into a quantum state.
The kernel value is the absolute squared overlap between two encoded
states.  Graph utilities are provided to build a fidelity‑based
adjacency graph from the intermediate states produced during a
feed‑forward pass.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pennylane as qml
import networkx as nx

# --------------------------------------------------------------------------- #
# 1. Quantum feature map
# --------------------------------------------------------------------------- #
class QuantumFeatureMap(qml.QNode):
    """Variational quantum circuit that encodes a classical vector into a state.

    The circuit consists of a layer of Ry rotations followed by a
    layer of CNOTs that entangle the qubits.  The parameters of the
    rotations are the data itself, making the circuit a feature map.
    """
    def __init__(self, device: qml.Device) -> None:
        super().__init__(device=device, interface="autograd")
        self.num_wires = device.num_wires

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for idx, val in enumerate(x):
            qml.RY(val, wires=idx)
        for i in range(self.num_wires - 1):
            qml.CNOT(wires=[i, i + 1])
        return qml.state()

# --------------------------------------------------------------------------- #
# 2. Quantum kernel
# --------------------------------------------------------------------------- #
class QuantumKernel:
    """Quantum kernel that evaluates the overlap between two encoded states."""
    def __init__(self, device: qml.Device) -> None:
        self.feature_map = QuantumFeatureMap(device)
        self.device = device

    def forward(self, x: np.ndarray, y: np.ndarray) -> float:
        state_x = self.feature_map(x)
        state_y = self.feature_map(y)
        return float(np.abs(state_x.conj().T @ state_y) ** 2)

# --------------------------------------------------------------------------- #
# 3. Unified kernel wrapper
# --------------------------------------------------------------------------- #
class UnifiedQuantumKernelGraph:
    """Hybrid kernel that can be used as a purely quantum kernel."""
    def __init__(self,
                 device: qml.Device,
                 mix_weight: float | None = None) -> None:
        self.quantum = QuantumKernel(device)
        self.mix = mix_weight if mix_weight is not None else 1.0

    def forward(self, x: np.ndarray, y: np.ndarray) -> float:
        return self.quantum.forward(x, y)

# --------------------------------------------------------------------------- #
# 4. Kernel matrix
# --------------------------------------------------------------------------- #
def compute_kernel_matrix(a: Sequence[np.ndarray],
                          b: Sequence[np.ndarray],
                          device: qml.Device) -> np.ndarray:
    kernel = UnifiedQuantumKernelGraph(device)
    return np.array([[kernel.forward(x, y) for y in b] for x in a])

# --------------------------------------------------------------------------- #
# 5. Graph utilities (adapted from GraphQNN)
# --------------------------------------------------------------------------- #
def random_unitary(num_qubits: int) -> np.ndarray:
    """Generate a random unitary matrix for ``num_qubits`` qubits."""
    dim = 2 ** num_qubits
    random_matrix = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    q, _ = np.linalg.qr(random_matrix)
    return q

def random_training_data(unitary: np.ndarray, samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    dataset: List[Tuple[np.ndarray, np.ndarray]] = []
    dim = unitary.shape[0]
    for _ in range(samples):
        state = np.random.randn(dim) + 1j * np.random.randn(dim)
        state /= np.linalg.norm(state)
        dataset.append((state, unitary @ state))
    return dataset

def random_network(qnn_arch: List[int],
                   samples: int) -> Tuple[List[int], List[List[np.ndarray]], List[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
    target_unitary = random_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: List[List[np.ndarray]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: List[np.ndarray] = []
        for _ in range(num_outputs):
            op = random_unitary(num_inputs + 1)
            if num_outputs > 1:
                op = np.kron(op, np.eye(2 ** (num_outputs - 1), dtype=complex))
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training_data, target_unitary

def _partial_trace(state: np.ndarray, keep: List[int]) -> np.ndarray:
    # Simplified partial trace: return the reduced state of the first qubit
    return state[:2]

def feedforward(qnn_arch: Sequence[int],
                unitaries: Sequence[Sequence[np.ndarray]],
                samples: Iterable[Tuple[np.ndarray, np.ndarray]]) -> List[List[np.ndarray]]:
    stored_states: List[List[np.ndarray]] = []
    for sample, _ in samples:
        layerwise: List[np.ndarray] = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = unitaries[layer][0] @ current_state
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states

def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(np.vdot(a, b)) ** 2)

def fidelity_adjacency(states: Sequence[np.ndarray],
                       threshold: float,
                       *,
                       secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

__all__ = [
    "UnifiedQuantumKernelGraph",
    "compute_kernel_matrix",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
