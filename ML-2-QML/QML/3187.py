"""GraphQNNGen119: quantum graph neural network + classifier utilities.

This module mirrors the classical API while using Qiskit and QuTiP for
state preparation, unitary evolution, and measurement.  It supports
side‑by‑side experimentation with the classical counterpart.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import qutip as qt
import scipy as sc
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

# ----- Core QNN utilities -----------------------------------------------------


def _tensored_id(num_qubits: int) -> qt.Qobj:
    """Identity operator on the full register."""
    identity = qt.qeye(2 ** num_qubits)
    dims = [2] * num_qubits
    identity.dims = [dims.copy(), dims.copy()]
    return identity


def _tensored_zero(num_qubits: int) -> qt.Qobj:
    """Zero projector on the full register."""
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
    amps = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
    amps /= sc.linalg.norm(amps)
    state = qt.Qobj(amps)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state


def random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
    dataset: List[Tuple[qt.Qobj, qt.Qobj]] = []
    n_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(n_qubits)
        dataset.append((state, unitary * state))
    return dataset


def random_network(qnn_arch: List[int], samples: int) -> Tuple[List[int], List[List[qt.Qobj]], List[Tuple[qt.Qobj, qt.Qobj]], qt.Qobj]:
    """Generate a random layered unitary network and training set."""
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: List[List[qt.Qobj]] = [[]]
    for layer in range(1, len(qnn_arch)):
        inp = qnn_arch[layer - 1]
        outp = qnn_arch[layer]
        ops: List[qt.Qobj] = []
        for _ in range(outp):
            op = _random_qubit_unitary(inp + 1)
            if outp > 1:
                op = qt.tensor(_random_qubit_unitary(inp + 1), _tensored_id(outp - 1))
                op = _swap_registers(op, inp, inp + _)
            ops.append(op)
        unitaries.append(ops)

    return qnn_arch, unitaries, training_data, target_unitary


def _partial_trace_keep(state: qt.Qobj, keep: Sequence[int]) -> qt.Qobj:
    if len(keep)!= len(state.dims[0]):
        return state.ptrace(list(keep))
    return state


def _partial_trace_remove(state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
    keep = list(range(len(state.dims[0])))
    for r in sorted(remove, reverse=True):
        keep.pop(r)
    return _partial_trace_keep(state, keep)


def _layer_channel(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]], layer: int, input_state: qt.Qobj) -> qt.Qobj:
    inp = qnn_arch[layer - 1]
    outp = qnn_arch[layer]
    state = qt.tensor(input_state, _tensored_zero(outp))

    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary

    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(inp))


def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qt.Qobj]],
    samples: Iterable[Tuple[qt.Qobj, qt.Qobj]],
) -> List[List[qt.Qobj]]:
    """Forward propagate a batch of quantum states through the network."""
    results: List[List[qt.Qobj]] = []
    for sample, _ in samples:
        layerwise = [sample]
        current = sample
        for layer in range(1, len(qnn_arch)):
            current = _layer_channel(qnn_arch, unitaries, layer, current)
            layerwise.append(current)
        results.append(layerwise)
    return results


def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    """Return the squared overlap between pure states."""
    return abs((a.dag() * b)[0, 0]) ** 2


def fidelity_adjacency(
    states: Sequence[qt.Qobj],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a weighted graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, si), (j, sj) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(si, sj)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# ----- Quantum classifier factory --------------------------------------------


def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, List, List, List[SparsePauliOp]]:
    """Create a quantum variational classifier with explicit encoding."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, q in zip(encoding, range(num_qubits)):
        circuit.rx(param, q)

    idx = 0
    for _ in range(depth):
        for q in range(num_qubits):
            circuit.ry(weights[idx], q)
            idx += 1
        for q in range(num_qubits - 1):
            circuit.cz(q, q + 1)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)
    ]
    return circuit, list(encoding), list(weights), observables


class GraphQNNGen119:
    """Unified container mirroring the classical API for quantum experiments."""

    def __init__(self, arch: Sequence[int], depth: int = 3, samples: int = 100):
        self.arch = list(arch)
        self.depth = depth
        self.arch, self.unitaries, self.training_data, self.target = random_network(self.arch, samples)
        self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(len(arch[0]), depth)

    def feedforward(self, samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]) -> List[List[qt.Qobj]]:
        """Convenience wrapper delegating to the module‑level function."""
        return feedforward(self.arch, self.unitaries, samples)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[qt.Qobj],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

    @staticmethod
    def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
        return state_fidelity(a, b)


__all__ = [
    "GraphQNNGen119",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
    "build_classifier_circuit",
]
