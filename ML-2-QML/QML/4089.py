from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

# ----------------------------------------------------------------------
# Helper functions for identity and zero states
def _tensored_identity(num_qubits: int) -> Statevector:
    dim = 2 ** num_qubits
    return Statevector(np.eye(dim))

def _tensored_zero(num_qubits: int) -> Statevector:
    dim = 2 ** num_qubits
    vec = np.zeros(dim, dtype=complex)
    vec[0] = 1.0
    return Statevector(vec)

# ----------------------------------------------------------------------
# Swap two qubits in a circuit
def _swap_registers(circ: QuantumCircuit, source: int, target: int) -> QuantumCircuit:
    if source == target:
        return circ
    circ.swap(source, target)
    return circ

# ----------------------------------------------------------------------
# Random unitary for given number of qubits
def _random_qubit_unitary(num_qubits: int) -> Statevector:
    dim = 2 ** num_qubits
    matrix = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
    q, _ = np.linalg.qr(matrix)
    return Statevector(q)

# ----------------------------------------------------------------------
# Random pure state
def _random_qubit_state(num_qubits: int) -> Statevector:
    dim = 2 ** num_qubits
    vec = np.random.normal(size=dim) + 1j * np.random.normal(size=dim)
    vec /= np.linalg.norm(vec)
    return Statevector(vec)

# ----------------------------------------------------------------------
# Generate training data: (input_state, target_state)
def random_training_data(unitary: Statevector, samples: int) -> List[Tuple[Statevector, Statevector]]:
    data: List[Tuple[Statevector, Statevector]] = []
    for _ in range(samples):
        st = _random_qubit_state(unitary.num_qubits)
        data.append((st, unitary @ st))
    return data

# ----------------------------------------------------------------------
# Build a random quantum graph neural network architecture
def random_network(
    qnn_arch: List[int],
    samples: int,
) -> Tuple[List[int], List[List[QuantumCircuit]], List[Tuple[Statevector, Statevector]], Statevector]:
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    circuits: List[List[QuantumCircuit]] = [[]]
    for layer in range(1, len(qnn_arch)):
        in_q = qnn_arch[layer - 1]
        out_q = qnn_arch[layer]
        layer_circuits: List[QuantumCircuit] = []
        for out in range(out_q):
            qc = QuantumCircuit(in_q + 1)
            qc.append(RealAmplitudes(in_q + 1, reps=3), range(in_q + 1))
            # swap test to encode output into ancillary qubit (index in_q)
            qc.h(in_q)
            qc.cswap(in_q, out, in_q + 1)
            qc.h(in_q)
            layer_circuits.append(qc)
        circuits.append(layer_circuits)

    return qnn_arch, circuits, training_data, target_unitary

# ----------------------------------------------------------------------
# Partial trace helpers
def _partial_trace_keep(state: Statevector, keep: Sequence[int]) -> Statevector:
    return state.ptrace(list(keep))

def _partial_trace_remove(state: Statevector, remove: Sequence[int]) -> Statevector:
    keep = list(range(state.num_qubits))
    for idx in sorted(remove, reverse=True):
        keep.pop(idx)
    return _partial_trace_keep(state, keep)

# ----------------------------------------------------------------------
# Layer channel: apply a layer of circuits and trace out ancilla qubits
def _layer_channel(
    qnn_arch: Sequence[int],
    circuits: Sequence[Sequence[QuantumCircuit]],
    layer: int,
    input_state: Statevector,
) -> Statevector:
    in_q = qnn_arch[layer - 1]
    out_q = qnn_arch[layer]
    # start from input state tensor with ancilla zeros
    ancilla = _tensored_zero(out_q)
    state = input_state.tensor(ancilla)
    # apply all circuits in this layer sequentially
    for circ in circuits[layer]:
        state = circ.evolve(state)
    # trace out ancilla qubits that were used as outputs
    return _partial_trace_remove(state, range(in_q, in_q + out_q))

# ----------------------------------------------------------------------
# Feedforward propagation
def feedforward(
    qnn_arch: Sequence[int],
    circuits: Sequence[Sequence[QuantumCircuit]],
    samples: Iterable[Tuple[Statevector, Statevector]],
) -> List[List[Statevector]]:
    results: List[List[Statevector]] = []
    for sample, _ in samples:
        layerwise: List[Statevector] = [sample]
        current = sample
        for layer in range(1, len(qnn_arch)):
            current = _layer_channel(qnn_arch, circuits, layer, current)
            layerwise.append(current)
        results.append(layerwise)
    return results

# ----------------------------------------------------------------------
# Fidelity between two pure states
def state_fidelity(a: Statevector, b: Statevector) -> float:
    return abs((a.dag() @ b)[0]) ** 2

# ----------------------------------------------------------------------
# Build a weighted graph from state fidelities
def fidelity_graph(
    states: Sequence[Statevector],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s_i, s_j)
        if fid >= threshold:
            G.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            G.add_edge(i, j, weight=secondary_weight)
    return G

# ----------------------------------------------------------------------
# Fast estimator for quantum circuits
class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized quantum circuit."""
    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

class FastEstimator(FastBaseEstimator):
    """Adds Gaussian shot noise to deterministic expectation values."""
    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in raw:
            noisy.append([rng.normal(c.real, max(1e-6, 1 / shots)) + 1j * rng.normal(c.imag, max(1e-6, 1 / shots)) for c in row])
        return noisy

__all__ = [
    "feedforward",
    "fidelity_graph",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "FastBaseEstimator",
    "FastEstimator",
]
