"""GraphQNN: quantum‑centric implementation.

This module mirrors the classical utilities but operates on quantum
states and unitaries.  It incorporates a variational circuit that
acts as a fully‑connected layer (FCL) and a simple quantum kernel
evaluator built with Qiskit.  The code is intentionally lightweight
so that it can be used as a drop‑in replacement for the original
seed, while providing a clear separation between classical and
quantum logic.

The design follows a *combination* scaling paradigm: the same API
is available for both classical and quantum back‑ends, and the
quantum methods expose variational circuits and a quantum kernel
that can be evaluated on a simulator.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
import qiskit
import qutip as qt
import scipy as sc

Tensor = qt.Qobj

# --------------------------------------------------------------------------- #
#  Fully‑connected layer – quantum version (inspired by FCL.py)
# --------------------------------------------------------------------------- #
class QuantumFCL:
    """Parameterised 1‑qubit circuit that mimics the classical FCL.

    The circuit consists of a Hadamard layer, a parameterised Ry gate,
    and a measurement.  The expectation value of the computational
    basis outcome is returned.
    """

    def __init__(self, backend: qiskit.providers.BaseBackend, shots: int = 1024):
        self._circuit = qiskit.QuantumCircuit(1)
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(0)
        self._circuit.barrier()
        self._circuit.ry(self.theta, 0)
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        result = job.result().get_counts(self._circuit)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        probabilities = counts / self.shots
        expectation = np.sum(states * probabilities)
        return np.array([expectation])


# --------------------------------------------------------------------------- #
#  Quantum kernel – simple variational circuit (inspired by QuantumKernelMethod.py)
# --------------------------------------------------------------------------- #
class QuantumKernel:
    """Variational kernel implemented with Qiskit.

    The kernel uses a single‑qubit Ry rotation for each input vector.
    It returns the absolute overlap between the two resulting states.
    """

    def __init__(self, backend: qiskit.providers.BaseBackend = None, shots: int = 1024):
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots

    def _state_from_vector(self, vec: np.ndarray) -> np.ndarray:
        """Encode a 1‑dimensional vector into a quantum state."""
        theta = 2 * np.arcsin(np.clip(vec[0], -1, 1))
        circ = qiskit.QuantumCircuit(1)
        circ.ry(theta, 0)
        circ.save_statevector()
        job = qiskit.execute(circ, self.backend, shots=1)
        state = job.result().get_statevector(circ)
        return state

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        sx = self._state_from_vector(x)
        sy = self._state_from_vector(y)
        overlap = np.abs(np.dot(sx.conj(), sy)) ** 2
        return overlap

    def kernel_matrix(self, a: Sequence[np.ndarray], b: Sequence[np.ndarray]) -> np.ndarray:
        """Compute the Gram matrix between two collections of vectors."""
        return np.array([[self(x, y) for y in b] for x in a])


# --------------------------------------------------------------------------- #
#  Core Graph‑QNN utilities – adapted from GraphQNN.py
# --------------------------------------------------------------------------- #
def _tensored_id(num_qubits: int) -> qt.Qobj:
    identity = qt.qeye(2 ** num_qubits)
    dims = [2] * num_qubits
    identity.dims = [dims.copy(), dims.copy()]
    return identity


def _tensored_zero(num_qubits: int) -> qt.Qobj:
    projector = qt.fock(2 ** num_qubits).proj()
    dims = [2] * num_qubits
    projector.dims = [dims.copy(), dims.copy()]
    return projector


def _swap_registers(op: qt.Qobj, source: int, target: int) -> qt.Qobj:
    if source == target:
        return op
    order = list(range(len(op.dims[0])))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)


def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    matrix = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    unitary = sc.linalg.orth(matrix)
    qobj = qt.Qobj(unitary)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj


def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    amplitudes = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
    amplitudes /= sc.linalg.norm(amplitudes)
    state = qt.Qobj(amplitudes)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state


def random_training_data(unitary: qt.Qobj, samples: int) -> list[tuple[qt.Qobj, qt.Qobj]]:
    dataset: list[tuple[qt.Qobj, qt.Qobj]] = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset


def random_network(qnn_arch: list[int], samples: int):
    """Generate a random layer‑wise unitary network and training data."""
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: list[list[qt.Qobj]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: list[qt.Qobj] = []
        for output in range(num_outputs):
            op = _random_qubit_unitary(num_inputs + 1)
            if num_outputs > 1:
                op = qt.tensor(_random_qubit_unitary(num_inputs + 1), _tensored_id(num_outputs - 1))
                op = _swap_registers(op, num_inputs, num_inputs + output)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training_data, target_unitary


def _partial_trace_keep(state: qt.Qobj, keep: Sequence[int]) -> qt.Qobj:
    if len(keep)!= len(state.dims[0]):
        return state.ptrace(list(keep))
    return state


def _partial_trace_remove(state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
    keep = list(range(len(state.dims[0])))
    for index in sorted(remove, reverse=True):
        keep.pop(index)
    return _partial_trace_keep(state, keep)


def _layer_channel(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qt.Qobj]],
    layer: int,
    input_state: qt.Qobj,
) -> qt.Qobj:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qt.tensor(input_state, _tensored_zero(num_outputs))

    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary

    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_inputs))


def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qt.Qobj]],
    samples: Iterable[tuple[qt.Qobj, qt.Qobj]],
) -> List[List[qt.Qobj]]:
    """Propagate a batch of quantum states through the unitary network."""
    stored_states: List[List[qt.Qobj]] = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states


def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    """Absolute squared overlap between two pure states."""
    return abs((a.dag() * b)[0, 0]) ** 2


def fidelity_adjacency(
    states: Sequence[qt.Qobj],
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


__all__ = [
    "QuantumFCL",
    "QuantumKernel",
    "random_network",
    "feedforward",
    "fidelity_adjacency",
    "random_training_data",
    "state_fidelity",
]
