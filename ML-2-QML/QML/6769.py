"""QuantumGraphEstimator – quantum implementation.

This module mirrors the structure of the classical estimator but
operates on a parameterised quantum circuit.  It supports
expectation‑value evaluation, shot‑noise simulation, and a
fidelity‑based adjacency graph of the state vectors produced after
each layer.  The public API matches the classical version so that
downstream code can switch between the two back‑ends without
modification.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Tuple, Union, Callable

import numpy as np
import itertools
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, RandomUnitary
from qiskit.quantum_info.operators.base_operator import BaseOperator
import networkx as nx

ScalarObservable = Union[BaseOperator, Callable[[Statevector], complex]]


def _ensure_batch(values: Sequence[float]) -> List[float]:
    # For quantum circuits, parameters are passed as lists of floats.
    return list(values)


class FastBaseEstimator:
    """
    Evaluate expectation values of observables for a parametrised circuit.

    The estimator accepts a Qiskit ``QuantumCircuit`` with symbolic
    parameters.  The circuit is bound to each parameter set before the
    statevector is generated.
    """

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
        """Compute expectation values for every parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(observable) for observable in observables]
            results.append(row)
        return results


class FastEstimator(FastBaseEstimator):
    """
    Wraps FastBaseEstimator to add shot‑noise simulation.

    The noise model is Gaussian with zero mean and a standard deviation
    equal to the inverse square root of the number of shots.
    """

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
            noisy_row = [rng.normal(0, max(1e-6, 1 / np.sqrt(shots))) + 1j * rng.normal(0, max(1e-6, 1 / np.sqrt(shots))) for _ in row]
            noisy.append(noisy_row)
        return noisy

    def fidelity_graph(
        self,
        parameter_sets: Sequence[Sequence[float]],
        *,
        threshold: float = 0.9,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """
        Build a weighted graph from the fidelities of the statevectors
        produced after each layer across all parameter sets.
        """
        states: List[Statevector] = []
        for values in parameter_sets:
            bound = self._bind(values)
            current_state = Statevector.from_instruction(bound)
            states.append(current_state)
        return fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)


# --- Graph‑based utilities (from GraphQNN) ---------------------------------

def random_training_data(unitary: Statevector, samples: int) -> List[Tuple[Statevector, Statevector]]:
    dataset: List[Tuple[Statevector, Statevector]] = []
    for _ in range(samples):
        state = Statevector.random(unitary.num_qubits)
        dataset.append((state, unitary @ state))
    return dataset


def random_network(qnn_arch: List[int], samples: int):
    num_qubits = qnn_arch[-1]
    target_unitary = RandomUnitary(num_qubits).data
    training_data = random_training_data(Statevector.from_label('0' * num_qubits), samples)

    # Build a simple random circuit with the same architecture
    circuit = QuantumCircuit(num_qubits)
    for layer in range(1, len(qnn_arch)):
        for _ in range(qnn_arch[layer]):
            circuit.h(range(num_qubits))
            circuit.cx(0, 1)
    return qnn_arch, circuit, training_data, target_unitary


def feedforward(
    qnn_arch: Sequence[int],
    circuit: QuantumCircuit,
    samples: Iterable[Tuple[Statevector, Statevector]],
) -> List[List[Statevector]]:
    stored_states: List[List[Statevector]] = []
    for sample, _ in samples:
        current_state = sample
        layerwise: List[Statevector] = [current_state]
        for _ in range(1, len(qnn_arch)):
            # Apply the whole circuit for illustration; in practice
            # one would slice the circuit per layer.
            current_state = Statevector.from_instruction(circuit)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states


def state_fidelity(a: Statevector, b: Statevector) -> float:
    """Return the absolute squared overlap between two statevectors."""
    return abs((a.dag() @ b)[0, 0]) ** 2


def fidelity_adjacency(
    states: Sequence[Statevector],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
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


__all__ = [
    "FastBaseEstimator",
    "FastEstimator",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
