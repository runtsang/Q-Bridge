"""UnifiedQMLEstimator – quantum‑only implementation using Qiskit and qutip."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Iterable as IterableType, List, Optional

import itertools
import networkx as nx
import qiskit
from qiskit import QuantumCircuit, Aer
from qiskit.quantum_info import Statevector
import qutip as qt
import numpy as np

# --------------------------------------------------------------------------- #
# Core quantum estimator – fast expectation evaluation
# --------------------------------------------------------------------------- #

class FastBaseEstimator:
    """Evaluate expectations for a parametrized quantum circuit."""
    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, param_values: Sequence[float]) -> QuantumCircuit:
        if len(param_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch")
        mapping = dict(zip(self._parameters, param_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: IterableType[qt.PauliOperator | qt.Qobj],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Return list of [obs1, *] per parameter set."""
        results: List[List[complex]] = []
        for params in parameter_sets:
            circ = self._bind(params)
            state = Statevector.from_instruction(circ)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def get_statevector(self, param_values: Sequence[float]) -> qt.Qobj:
        circ = self._bind(param_values)
        return Statevector.from_instruction(circ).data

class FastEstimator(FastBaseEstimator):
    """Add optional shot noise by sampling from the measurement distribution."""
    def evaluate(
        self,
        observables: IterableType[qt.PauliOperator | qt.Qobj],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[List[complex]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in raw:
            noisy_row = []
            for val in row:
                # Convert expectation to probability of measuring +1 for Z
                prob_plus = (1 + float(val.real)) / 2
                counts = rng.binomial(shots, prob_plus)
                noisy_val = 2 * counts / shots - 1
                noisy_row.append(complex(noisy_val, 0))
            noisy.append(noisy_row)
        return noisy

# --------------------------------------------------------------------------- #
# Utility functions for graph‑based quantum neural networks
# --------------------------------------------------------------------------- #

def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    matrix = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
    unitary = np.linalg.qr(matrix)[0]
    return qt.Qobj(unitary)

def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    vec = np.random.normal(size=(dim, 1)) + 1j * np.random.normal(size=(dim, 1))
    vec /= np.linalg.norm(vec)
    return qt.Qobj(vec)

def random_training_data(unitary: qt.Qobj, samples: int) -> List[tuple[qt.Qobj, qt.Qobj]]:
    dataset = []
    for _ in range(samples):
        state = _random_qubit_state(num_qubits=int(np.log2(unitary.shape[0])))
        dataset.append((state, unitary * state))
    return dataset

def random_network(qnn_arch: List[int], samples: int):
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: List[List[qt.Qobj]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: List[qt.Qobj] = []
        for output in range(num_outputs):
            op = _random_qubit_unitary(num_inputs + 1)
            if num_outputs > 1:
                op = qt.tensor(_random_qubit_unitary(num_inputs + 1), _random_qubit_unitary(num_outputs - 1))
                # Swap the output qubit into the last position
                perm = list(range(num_inputs + num_outputs))
                perm[num_inputs + output], perm[-1] = perm[-1], perm[num_inputs + output]
                op = op.permute(perm)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training_data, target_unitary

def _partial_trace_keep(state: qt.Qobj, keep: Sequence[int]) -> qt.Qobj:
    if len(keep)!= len(state.dims[0]):
        return state.ptrace(list(keep))
    return state

def _partial_trace_remove(state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
    keep = list(range(len(state.dims[0])))
    for idx in sorted(remove, reverse=True):
        keep.pop(idx)
    return _partial_trace_keep(state, keep)

def _layer_channel(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]], layer: int, input_state: qt.Qobj) -> qt.Qobj:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qt.tensor(input_state, _random_qubit_state(num_outputs))
    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary
    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_inputs))

def feedforward(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]], samples: Iterable[tuple[qt.Qobj, qt.Qobj]]):
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
    """Return the absolute squared overlap between pure states a and b."""
    return abs((a.dag() * b)[0, 0]) ** 2

def fidelity_adjacency(
    states: Sequence[qt.Qobj],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            G.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            G.add_edge(i, j, weight=secondary_weight)
    return G

# --------------------------------------------------------------------------- #
# Hybrid estimator that simply forwards to FastEstimator
# --------------------------------------------------------------------------- #

class UnifiedQMLEstimator(FastEstimator):
    """Quantum‑only estimator that inherits FastEstimator functionality."""
    def __init__(self, circuit: QuantumCircuit) -> None:
        super().__init__(circuit)

__all__ = [
    "FastBaseEstimator",
    "FastEstimator",
    "UnifiedQMLEstimator",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
