"""GraphQNNGen: Quantum graph neural network with estimator utilities.

This module merges the quantum seed with FastBaseEstimator for
parametrized circuits.  The API parallels the classical version but
uses qutip for state propagation and qiskit for expectation‑value
evaluation.  It supports random unitary generation, feed‑forward
propagation through a layered circuit, fidelity‑based graph
construction, and a lightweight estimator that can add shot noise.

Typical usage:

    net = GraphQNNGen([2, 3])
    arch, ops, data, target = net.random_network(samples=5)
    states = net.feedforward(ops, data)
    graph = net.fidelity_adjacency([s[-1] for s in states], 0.8)
    est = net.FastEstimator()
    results = est.evaluate([qiskit.quantum_info.sparse_pauli_matrix('Z' * 2)], [[0.1, 0.2]])
"""

from __future__ import annotations

import itertools
from typing import List, Tuple, Sequence, Iterable, Optional

import networkx as nx
import numpy as np
import qutip as qt
import scipy as sc
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

Tensor = qt.Qobj


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


def random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
    dataset: List[Tuple[qt.Qobj, qt.Qobj]] = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset


class GraphQNNGen:
    """Quantum graph‑based neural network with estimator utilities."""

    def __init__(self, arch: Sequence[int]) -> None:
        self.arch = list(arch)

    def random_network(
        self, samples: int
    ) -> Tuple[List[int], List[List[qt.Qobj]], List[Tuple[qt.Qobj, qt.Qobj]], qt.Qobj]:
        target_unitary = _random_qubit_unitary(self.arch[-1])
        training_data = random_training_data(target_unitary, samples)
        unitaries: List[List[qt.Qobj]] = [[]]
        for layer in range(1, len(self.arch)):
            num_inputs = self.arch[layer - 1]
            num_outputs = self.arch[layer]
            layer_ops: List[qt.Qobj] = []
            for output in range(num_outputs):
                op = _random_qubit_unitary(num_inputs + 1)
                if num_outputs > 1:
                    op = qt.tensor(_random_qubit_unitary(num_inputs + 1), _tensored_id(num_outputs - 1))
                    op = _swap_registers(op, num_inputs, num_inputs + output)
                layer_ops.append(op)
            unitaries.append(layer_ops)
        return self.arch, unitaries, training_data, target_unitary

    def _partial_trace_keep(self, state: qt.Qobj, keep: Sequence[int]) -> qt.Qobj:
        if len(keep)!= len(state.dims[0]):
            return state.ptrace(list(keep))
        return state

    def _partial_trace_remove(self, state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
        keep = list(range(len(state.dims[0])))
        for index in sorted(remove, reverse=True):
            keep.pop(index)
        return self._partial_trace_keep(state, keep)

    def _layer_channel(
        self,
        layer: int,
        input_state: qt.Qobj,
        unitaries: Sequence[Sequence[qt.Qobj]],
    ) -> qt.Qobj:
        num_inputs = self.arch[layer - 1]
        num_outputs = self.arch[layer]
        state = qt.tensor(input_state, _tensored_zero(num_outputs))
        layer_unitary = unitaries[layer][0].copy()
        for gate in unitaries[layer][1:]:
            layer_unitary = gate * layer_unitary
        return self._partial_trace_remove(
            layer_unitary * state * layer_unitary.dag(), range(num_inputs)
        )

    def feedforward(
        self, unitaries: Sequence[Sequence[qt.Qobj]], samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]
    ) -> List[List[qt.Qobj]]:
        stored_states: List[List[qt.Qobj]] = []
        for sample, _ in samples:
            layerwise = [sample]
            current_state = sample
            for layer in range(1, len(self.arch)):
                current_state = self._layer_channel(layer, current_state, unitaries)
                layerwise.append(current_state)
            stored_states.append(layerwise)
        return stored_states

    @staticmethod
    def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
        return abs((a.dag() * b)[0, 0]) ** 2

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[qt.Qobj],
        threshold: float,
        *,
        secondary: Optional[float] = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNNGen.state_fidelity(s_i, s_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # Estimator utilities ----------------------------------------------------
    class FastBaseEstimator:
        """Evaluate expectation values of observables for a parametrized circuit."""

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
            observables = list(observables)
            results: List[List[complex]] = []
            for values in parameter_sets:
                state = Statevector.from_instruction(self._bind(values))
                row = [state.expectation_value(obs) for obs in observables]
                results.append(row)
            return results

    class FastEstimator(FastBaseEstimator):
        """Adds Gaussian shot noise to deterministic estimates."""

        def evaluate(
            self,
            observables: Iterable[BaseOperator],
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
                noisy_row = [
                    rng.normal(val.real, max(1e-6, 1 / shots))
                    + 1j * rng.normal(val.imag, max(1e-6, 1 / shots))
                    for val in row
                ]
                noisy.append(noisy_row)
            return noisy


__all__ = ["GraphQNNGen"]
