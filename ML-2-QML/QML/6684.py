"""GraphQNNHybrid - quantum implementation using PennyLane.

Implements a quantum graph‑based neural network with random unitary layers,
fidelity utilities, and a regression dataset.  The class shares the same
interface as the classical version, allowing side‑by‑side hybrid experiments.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import networkx as nx

Array = np.ndarray
Tensor = torch.Tensor


def _random_qubit_unitary(num_qubits: int) -> Array:
    """Generate a random unitary on `num_qubits` qubits."""
    dim = 2 ** num_qubits
    mat = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    q, _ = np.linalg.qr(mat)  # QR gives a unitary
    return q


def _random_qubit_state(num_qubits: int) -> Array:
    """Generate a random pure state on `num_qubits` qubits."""
    dim = 2 ** num_qubits
    vec = np.random.randn(dim) + 1j * np.random.randn(dim)
    vec /= np.linalg.norm(vec)
    return vec


def random_training_data(unitary: Array, samples: int) -> List[Tuple[Array, Array]]:
    """Generate training pairs (state, U|state>)."""
    data = []
    num_qubits = int(np.log2(unitary.shape[0]))
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        data.append((state, unitary @ state))
    return data


def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random quantum network and training data."""
    num_wires = qnn_arch[-1]
    target_unitary = _random_qubit_unitary(num_wires)
    training_data = random_training_data(target_unitary, samples)
    # One unitary per layer (excluding input layer)
    unitaries = [_random_qubit_unitary(num_wires) for _ in range(len(qnn_arch) - 1)]
    return list(qnn_arch), unitaries, training_data, target_unitary


def _apply_unitaries(state: Array, ops: List[Array]) -> Array:
    """Apply a sequence of unitaries to a state vector."""
    out = state
    for op in ops:
        out = op @ out
    return out


def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Array],
    samples: Iterable[Tuple[Array, Array]],
) -> List[List[Array]]:
    """Propagate each sample through the quantum network."""
    outputs: List[List[Array]] = []
    for state, _ in samples:
        layer_out = [state]
        current = state
        for U in unitaries:
            current = _apply_unitaries(current, [U])
            layer_out.append(current)
        outputs.append(layer_out)
    return outputs


def state_fidelity(a: Array, b: Array) -> float:
    """Squared overlap between two pure states."""
    return abs(np.vdot(a, b)) ** 2


def fidelity_adjacency(
    states: Sequence[Array],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a weighted graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s_i, s_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


class HybridDatasetQuantum(torch.utils.data.Dataset):
    """Dataset yielding complex quantum states and regression targets."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = self._generate_data(num_wires, samples)

    @staticmethod
    def _generate_data(num_wires: int, samples: int) -> Tuple[Array, Array]:
        dim = 2 ** num_wires
        states = np.zeros((samples, dim), dtype=complex)
        labels = np.zeros(samples, dtype=np.float32)
        thetas = 2 * np.pi * np.random.rand(samples)
        phis = 2 * np.pi * np.random.rand(samples)
        for i in range(samples):
            state = np.zeros(dim, dtype=complex)
            state[0] = np.cos(thetas[i])
            state[-1] = np.exp(1j * phis[i]) * np.sin(thetas[i])
            states[i] = state
            labels[i] = np.sin(2 * thetas[i]) * np.cos(phis[i])
        return states, labels.astype(np.float32)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class GraphQNNHybridQuantum:
    """Quantum graph‑based neural network using PennyLane."""

    def __init__(self, arch: Sequence[int]):
        self.arch = list(arch)
        self.num_wires = arch[-1]
        # Random unitary per layer (excluding input)
        self.unitaries = [_random_qubit_unitary(self.num_wires) for _ in range(len(arch) - 1)]

    def random_network(self, samples: int):
        return random_network(self.arch, samples)

    def feedforward(
        self,
        unitaries: Sequence[Array],
        samples: Iterable[Tuple[Array, Array]],
    ) -> List[List[Array]]:
        return feedforward(self.arch, unitaries, samples)

    def build_fidelity_graph(
        self,
        states: Sequence[Array],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

    def forward(self, state_batch: Array) -> Array:
        """Apply the full network to a batch of states."""
        outputs = []
        for state in state_batch:
            current = state
            for U in self.unitaries:
                current = _apply_unitaries(current, [U])
            outputs.append(current)
        return np.array(outputs)


__all__ = [
    "GraphQNNHybridQuantum",
    "HybridDatasetQuantum",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
