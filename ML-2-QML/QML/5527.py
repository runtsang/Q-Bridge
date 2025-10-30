from __future__ import annotations

import itertools
import numpy as np
import qiskit
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.providers import Backend
from typing import Iterable, Sequence, List, Optional
import networkx as nx

def _state_fidelity(a: Statevector, b: Statevector) -> float:
    return abs((a.dag() * b)[0, 0]) ** 2

def _build_fidelity_graph(states: Sequence[Statevector], threshold: float,
                           *, secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = _state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

class FastBaseEstimatorGen:
    """Hybrid estimator that evaluates a parametrized quantum circuit with optional shot noise,
    fidelity graph construction, and convenient QNN utilities."""
    def __init__(self,
                 circuit: QuantumCircuit,
                 observables: Sequence[SparsePauliOp],
                 backend: Optional[Backend] = None,
                 shots: Optional[int] = None) -> None:
        self.circuit = circuit
        self._parameters = list(circuit.parameters)
        self.observables = list(observables)
        self.backend = backend or qiskit.Aer.get_backend("statevector_simulator")
        self.shots = shots
        self._estimator = StatevectorEstimator(backend=self.backend)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self,
                 parameter_sets: Sequence[Sequence[float]],
                 *,
                 shots: int | None = None,
                 seed: int | None = None) -> List[List[complex]]:
        if shots is None:
            shots = self.shots
        results: List[List[complex]] = []
        for params in parameter_sets:
            bound = self._bind(params)
            state = Statevector.from_instruction(bound)
            row = [state.expectation_value(obs) for obs in self.observables]
            results.append(row)
        if shots is not None:
            rng = np.random.default_rng(seed)
            for i, row in enumerate(results):
                noisy_row = [complex(rng.normal(val.real, max(1e-6, 1 / shots)),
                                     rng.normal(val.imag, max(1e-6, 1 / shots)))
                             for val in row]
                results[i] = noisy_row
        return results

    def fidelity_graph(self,
                       parameter_sets: Sequence[Sequence[float]],
                       threshold: float,
                       *,
                       secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
        statevectors = [Statevector.from_instruction(self._bind(params))
                        for params in parameter_sets]
        return _build_fidelity_graph(statevectors, threshold,
                                     secondary=secondary,
                                     secondary_weight=secondary_weight)

    @staticmethod
    def create_estimator_qnn(circuit: QuantumCircuit,
                             observables: SparsePauliOp,
                             input_params: Sequence[Parameter],
                             weight_params: Sequence[Parameter]) -> EstimatorQNN:
        return EstimatorQNN(circuit=circuit,
                            observables=observables,
                            input_params=input_params,
                            weight_params=weight_params,
                            estimator=StatevectorEstimator())

    @staticmethod
    def random_qnn_architecture(qnn_arch: Sequence[int], samples: int):
        from GraphQNN import random_network
        return random_network(qnn_arch, samples)

__all__ = ["FastBaseEstimatorGen"]
