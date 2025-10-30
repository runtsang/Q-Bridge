"""Unified estimator for parametrized quantum circuits with graph-based analysis.

This module mirrors the classical estimator but operates on Qiskit circuits.
It provides:
- batched evaluation of parameter sets and expectation values
- optional Gaussian shot noise to mimic measurement statistics
- fidelity graph construction from Statevector or Qobj states
- utilities to generate random quantum networks and training data
"""

from __future__ import annotations

from collections.abc import Iterable as IterableType, Sequence
from typing import Callable, List, Union

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

import networkx as nx


QuantumObservable = Callable[[Statevector], complex | float]
ArrayOrTensor = Union[Statevector, List[complex], Sequence[complex]]


def _ensure_batch(values: Sequence[float]) -> list[float]:
    """Return a list of floats; Qiskit circuits expect plain python lists for parameters."""
    return list(values)


class UnifiedQNNEstimator:
    """Evaluate a Qiskit quantum circuit for batches of parameters and observables."""

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, param_values: Sequence[float]) -> QuantumCircuit:
        if len(param_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, param_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: IterableType[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Return expectation values for each parameter set.

        If ``shots`` is supplied, Gaussian noise with variance 1/shots is added
        to each deterministic expectation value to emulate shot noise.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for param_vals in parameter_sets:
            state = Statevector.from_instruction(self._bind(param_vals))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in results:
            noisy_row = [
                rng.normal(mean.real, max(1e-6, 1 / shots)) + 1j * rng.normal(mean.imag, max(1e-6, 1 / shots))
                for mean in row
            ]
            noisy.append(noisy_row)
        return noisy

    # ------------------------------------------------------------------ #
    #  Graph-based fidelity utilities (quantum)
    # ------------------------------------------------------------------ #

    @staticmethod
    def state_fidelity(a: Statevector, b: Statevector) -> float:
        """Squared overlap between two pure statevectors."""
        return abs((a.dag() * b)[0, 0]) ** 2

    @staticmethod
    def fidelity_graph(
        states: Sequence[Statevector],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Return a weighted adjacency graph built from state fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for i, a in enumerate(states):
            for j, b in enumerate(states[i + 1 :], start=i + 1):
                fid = UnifiedQNNEstimator.state_fidelity(a, b)
                if fid >= threshold:
                    graph.add_edge(i, j, weight=1.0)
                elif secondary is not None and fid >= secondary:
                    graph.add_edge(i, j, weight=secondary_weight)
        return graph

    # ------------------------------------------------------------------ #
    #  Random network / training data helpers (toy quantum)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _random_qubit_unitary(num_qubits: int):
        dim = 2 ** num_qubits
        matrix = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
        unitary, _ = np.linalg.qr(matrix)
        return unitary

    @staticmethod
    def _random_qubit_state(num_qubits: int):
        dim = 2 ** num_qubits
        vec = np.random.normal(size=(dim, 1)) + 1j * np.random.normal(size=(dim, 1))
        vec /= np.linalg.norm(vec)
        return Statevector(vec)

    @staticmethod
    def random_training_data(unitary: np.ndarray, samples: int) -> List[tuple[Statevector, Statevector]]:
        dataset: List[tuple[Statevector, Statevector]] = []
        for _ in range(samples):
            state = UnifiedQNNEstimator._random_qubit_state(unitary.shape[0].bit_length() - 1)
            dataset.append((state, Statevector(unitary @ state.data)))
        return dataset

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int):
        # Build a toy network of random unitaries per layer
        unitaries: list[list[np.ndarray]] = [[]]
        for layer in range(1, len(qnn_arch)):
            num_inputs = qnn_arch[layer - 1]
            unitary = UnifiedQNNEstimator._random_qubit_unitary(num_inputs + 1)
            unitaries.append([unitary])
        target_unitary = UnifiedQNNEstimator._random_qubit_unitary(qnn_arch[-1])
        training_data = UnifiedQNNEstimator.random_training_data(target_unitary, samples)
        return list(qnn_arch), unitaries, training_data, target_unitary

    @staticmethod
    def feedforward(
        qnn_arch: Sequence[int],
        unitaries: Sequence[Sequence[np.ndarray]],
        samples: Iterable[tuple[Statevector, Statevector]],
    ) -> List[List[Statevector]]:
        stored: List[List[Statevector]] = []
        for sample, _ in samples:
            layerwise = [sample]
            current = sample
            for layer in range(1, len(qnn_arch)):
                unitary = unitaries[layer][0]
                current = Statevector(unitary @ current.data)
                layerwise.append(current)
            stored.append(layerwise)
        return stored


__all__ = ["UnifiedQNNEstimator"]
