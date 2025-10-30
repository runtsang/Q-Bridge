"""Hybrid estimator combining quantum circuit evaluation with graph utilities.

This module extends the original QML FastBaseEstimator to support
random network generation, feed‑forward state propagation, and
fidelity‑based graph construction.  It evaluates a Qiskit circuit
using state‑vector simulation and can add Gaussian shot noise to
simulate measurement statistics.
"""

from __future__ import annotations

from collections.abc import Iterable, List, Sequence
from typing import Optional

import itertools
import numpy as np

import networkx as nx
import qiskit
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

import qutip as qt

ScalarObservable = BaseOperator


class FastBaseEstimator:
    """Evaluate a Qiskit circuit for a set of parameters, with optional shot noise.

    Parameters
    ----------
    circuit : QuantumCircuit
        A parameterized quantum circuit.
    shots : int | None, optional
        If provided, Gaussian noise with variance 1/shots is added to each
        expectation to emulate measurement shot noise.
    seed : int | None, optional
        Seed for the random number generator used for shot noise.
    """

    def __init__(self,
                 circuit: QuantumCircuit,
                 *,
                 shots: Optional[int] = None,
                 seed: Optional[int] = None) -> None:
        self.circuit = circuit
        self.shots = shots
        self.seed = seed
        self._parameters = list(circuit.parameters)

    @staticmethod
    def _quantum_expectation(circuit: QuantumCircuit,
                             observables: Iterable[BaseOperator],
                             parameter_values: Sequence[float]) -> List[complex]:
        """Evaluate a state‑vector expectation for each observable."""
        if len(parameter_values)!= len(circuit.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(circuit.parameters, parameter_values))
        bound = circuit.assign_parameters(mapping, inplace=False)
        state = Statevector.from_instruction(bound)
        return [state.expectation_value(observable) for observable in observables]

    def evaluate(self,
                 observables: Iterable[BaseOperator],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        """Evaluate the circuit on the provided parameter sets."""
        raw = []
        for values in parameter_sets:
            row = self._quantum_expectation(self.circuit, observables, values)
            raw.append(row)
        if self.shots is None:
            return raw
        rng = np.random.default_rng(self.seed)
        noisy: List[List[complex]] = []
        for row in raw:
            noisy_row = [rng.normal(val.real, max(1e-6, 1 / self.shots)) + 1j * rng.normal(val.imag, max(1e-6, 1 / self.shots))
                         for val in row]
            noisy.append(noisy_row)
        return noisy


# --------------------------------------------------------------------------- #
#  Graph‑based utilities for quantum networks
# --------------------------------------------------------------------------- #

def _tensored_id(num_qubits: int) -> qt.Qobj:
    """Return an identity operator on `num_qubits` qubits."""
    identity = qt.qeye(2 ** num_qubits)
    dims = [2] * num_qubits
    identity.dims = [dims.copy(), dims.copy()]
    return identity


def _tensored_zero(num_qubits: int) -> qt.Qobj:
    """Return a projector onto |0...0>."""
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
    import scipy as sc
    dim = 2 ** num_qubits
    matrix = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    unitary = sc.linalg.orth(matrix)
    qobj = qt.Qobj(unitary)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj


def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    import scipy as sc
    dim = 2 ** num_qubits
    amplitudes = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
    amplitudes /= sc.linalg.norm(amplitudes)
    state = qt.Qobj(amplitudes)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state


def random_training_data(unitary: qt.Qobj, samples: int) -> list[tuple[qt.Qobj, qt.Qobj]]:
    """Generate training pairs (state, target) for a given unitary."""
    dataset = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset


def random_network(qnn_arch: list[int], samples: int):
    """Create a random QNN architecture with unitaries and training data."""
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


def _layer_channel(qnn_arch: Sequence[int],
                   unitaries: Sequence[Sequence[qt.Qobj]],
                   layer: int,
                   input_state: qt.Qobj) -> qt.Qobj:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qt.tensor(input_state, _tensored_zero(num_outputs))

    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary

    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_inputs))


def feedforward(qnn_arch: Sequence[int],
                unitaries: Sequence[Sequence[qt.Qobj]],
                samples: Iterable[tuple[qt.Qobj, qt.Qobj]]):
    """Propagate states through a QNN and record intermediate states."""
    stored_states = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states


def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    """Return the absolute squared overlap between pure states."""
    return abs((a.dag() * b)[0, 0]) ** 2


def fidelity_adjacency(states: Sequence[qt.Qobj],
                       threshold: float,
                       *,
                       secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    """Create a weighted adjacency graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


__all__ = ["FastBaseEstimator",
           "feedforward",
           "fidelity_adjacency",
           "random_network",
           "random_training_data",
           "state_fidelity"]
