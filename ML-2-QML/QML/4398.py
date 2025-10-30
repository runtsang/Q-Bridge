"""GraphQNN module for quantum machine learning.

This module implements a quantum graph neural network that mirrors the
classical interface but uses Qiskit to construct and evaluate
parameterized quantum circuits.  It also provides regression and
kernel utilities that operate on quantum states.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import networkx as nx
import qiskit
from qiskit import Aer, execute, QuantumCircuit
from qiskit.quantum_info import Statevector, Operator


Tensor = torch.Tensor


def _random_qubit_unitary(num_qubits: int) -> Statevector:
    """Return a random unitary as a Statevector object."""
    dim = 2 ** num_qubits
    matrix = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
    q, _ = np.linalg.qr(matrix)
    return Statevector(q)


def _random_qubit_state(num_qubits: int) -> Statevector:
    """Return a random pure state."""
    dim = 2 ** num_qubits
    vec = np.random.normal(size=(dim, 1)) + 1j * np.random.normal(size=(dim, 1))
    vec /= np.linalg.norm(vec)
    return Statevector(vec)


def random_training_data(unitary: Statevector, samples: int) -> List[Tuple[Statevector, Statevector]]:
    """Generate (state, target) pairs by applying a fixed unitary."""
    data: List[Tuple[Statevector, Statevector]] = []
    for _ in range(samples):
        state = _random_qubit_state(unitary.num_qubits)
        data.append((state, unitary.evolve(state)))
    return data


def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random quantum graph neural network."""
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: List[List[Statevector]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: List[Statevector] = []
        for output in range(num_outputs):
            op = _random_qubit_unitary(num_inputs + 1)
            if num_outputs > 1:
                op = _random_qubit_unitary(num_inputs + 1)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return list(qnn_arch), unitaries, training_data, target_unitary


def _partial_trace_keep(state: Statevector, keep: Sequence[int]) -> Statevector:
    """Return the reduced state on the specified qubits."""
    return state.ptrace(list(keep))


def _partial_trace_remove(state: Statevector, remove: Sequence[int]) -> Statevector:
    """Return the reduced state on the remaining qubits."""
    keep = list(range(state.num_qubits))
    for idx in sorted(remove, reverse=True):
        keep.pop(idx)
    return _partial_trace_keep(state, keep)


def _layer_channel(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[Statevector]],
    layer: int,
    input_state: Statevector,
) -> Statevector:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    # Pad input with zeros for additional output qubits
    padded = np.concatenate([input_state.data, np.zeros((2 ** num_outputs, 1), dtype=complex)])
    state = Statevector(padded)
    # Compose unitary
    layer_unitary = unitaries[layer][0].data
    for gate in unitaries[layer][1:]:
        layer_unitary = gate.data @ layer_unitary
    new_state = Statevector(layer_unitary @ state.data)
    return _partial_trace_remove(new_state, range(num_inputs))


def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[Statevector]],
    samples: Iterable[Tuple[Statevector, Statevector]],
) -> List[List[Statevector]]:
    """Propagate quantum states through the network."""
    stored_states: List[List[Statevector]] = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states


def state_fidelity(a: Statevector, b: Statevector) -> float:
    """Squared overlap between two pure states."""
    return abs(a.fidelity(b)) ** 2


def fidelity_adjacency(
    states: Sequence[Statevector],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a weighted graph from state fidelities."""
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for (i, si), (j, sj) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(si, sj)
        if fid >= threshold:
            G.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            G.add_edge(i, j, weight=secondary_weight)
    return G


# ----------------------------------------------------------------------
# Regression utilities
# ----------------------------------------------------------------------
def generate_superposition_data(num_wires: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sample states of the form cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>."""
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)


class RegressionDataset(torch.utils.data.Dataset):
    """Dataset yielding quantum states and regression targets."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class QModel(nn.Module):
    """Simple quantum regression model using a variational circuit."""

    def __init__(self, num_wires: int):
        super().__init__()
        self.num_wires = num_wires
        # Trainable rotation angles for each qubit
        self.params = nn.Parameter(torch.randn(num_wires))
        self.linear = nn.Linear(num_wires, 1)
        self.backend = Aer.get_backend("statevector_simulator")

    def forward(self, states: Tensor) -> Tensor:
        """Compute a prediction for a batch of state vectors."""
        batch_size = states.shape[0]
        features = []
        for i in range(batch_size):
            # Build a circuit that prepares the input state
            circ = QuantumCircuit(self.num_wires)
            # Encode the state as amplitudes using a basis rotation
            amp = states[i].real
            for q in range(self.num_wires):
                circ.ry(amp[q], q)
            # Variational layer
            for q in range(self.num_wires):
                circ.rx(self.params[q], q)
            # Simulate
            result = execute(circ, self.backend, shots=1).result()
            state = Statevector(result.get_statevector(circ))
            # Expectation of PauliZ on each qubit
            exp = []
            for q in range(self.num_wires):
                op = Operator("Z")
                for _ in range(self.num_wires - 1):
                    op = op.tensor(Operator("I"))
                exp.append(state.expectation_value(op).real)
            features.append(exp)
        features = torch.tensor(features, dtype=torch.float32)
        return self.linear(features).squeeze(-1)


# ----------------------------------------------------------------------
# Kernel utilities
# ----------------------------------------------------------------------
class KernalAnsatz:
    """Quantum kernel that computes the overlap of two state vectors."""

    def __init__(self):
        pass

    def __call__(self, x: Statevector, y: Statevector) -> float:
        return abs(x.fidelity(y)) ** 2


class Kernel:
    """Wrapper that exposes a single scalar kernel value."""

    def __call__(self, x: Statevector, y: Statevector) -> float:
        return KernalAnsatz()(x, y)


def kernel_matrix(a: Sequence[Statevector], b: Sequence[Statevector]) -> np.ndarray:
    """Compute the Gram matrix between two sets of quantum states."""
    kernel = Kernel()
    return np.array([[kernel(x, y) for y in b] for x in a])


class GraphQNN:
    """Quantum graph neural network wrapper."""

    def __init__(self, architecture: Sequence[int]):
        self.architecture = list(architecture)

    @staticmethod
    def random_network(arch: Sequence[int], samples: int):
        return random_network(arch, samples)

    @staticmethod
    def feedforward(
        arch: Sequence[int],
        unitaries: Sequence[Sequence[Statevector]],
        samples: Iterable[Tuple[Statevector, Statevector]],
    ):
        return feedforward(arch, unitaries, samples)

    @staticmethod
    def state_fidelity(a: Statevector, b: Statevector) -> float:
        return state_fidelity(a, b)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Statevector],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)


__all__ = [
    "GraphQNN",
]
