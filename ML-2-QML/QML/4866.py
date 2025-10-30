"""Hybrid quantum estimator that extends FastBaseEstimator with graph utilities and a sampler.

The estimator accepts a parameterised :class:`QuantumCircuit` and provides:
* evaluation of expectation values for a list of observables.
* generation of random quantum networks and training data.
* construction of a fidelity‑based adjacency graph from the state vectors.
* construction of a quantum sampler network for generative modelling.
"""

from __future__ import annotations

import networkx as nx
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from collections.abc import Iterable, Sequence

# Local utilities from the original seed
from.FastBaseEstimator import FastBaseEstimator
from.GraphQNN import (
    random_network as quantum_random_network,
    feedforward as quantum_feedforward,
    fidelity_adjacency as quantum_fidelity_adjacency,
)
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN


class HybridEstimator(FastBaseEstimator):
    """Hybrid quantum estimator that extends FastBaseEstimator with graph utilities and a sampler."""

    def __init__(self, circuit: QuantumCircuit) -> None:
        super().__init__(circuit)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> list[list[complex]]:
        """Compute expectation values for each parameter set and observable."""
        return super().evaluate(observables, parameter_sets)

    @staticmethod
    def random_network(
        arch: list[int], samples: int
    ) -> tuple[list[int], list[list], list[tuple], object]:
        """Return a random unitary network and a training set of state pairs."""
        return quantum_random_network(arch, samples)

    @staticmethod
    def feedforward(
        arch: list[int],
        unitaries: list[list],
        samples: Iterable[tuple],
    ) -> list[list]:
        """Propagate all samples through the quantum network."""
        return quantum_feedforward(arch, unitaries, samples)

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[object],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Build a weighted graph from state fidelities."""
        return quantum_fidelity_adjacency(states, threshold, secondary=secondary, secondary_weight=secondary_weight)

    @staticmethod
    def sampler_network() -> QSamplerQNN:
        """Return a parameterised quantum sampler circuit."""
        return QSamplerQNN()


__all__ = ["HybridEstimator"]
