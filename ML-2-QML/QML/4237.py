import numpy as np
import networkx as nx
import itertools
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info import Operator
from typing import Iterable, List, Sequence
from.FastBaseEstimator import FastBaseEstimator

class HybridEstimator(FastBaseEstimator):
    """
    Quantum hybrid estimator that evaluates expectation values of a
    parametrized circuit, injects shot noise, and constructs a
    fidelity‑based adjacency graph from the resulting quantum states.
    """

    def __init__(self, circuit: QuantumCircuit):
        super().__init__(circuit)
        self._parameters = list(circuit.parameters)

    def _bind(self, param_values: Sequence[float]) -> QuantumCircuit:
        if len(param_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch.")
        mapping = dict(zip(self._parameters, param_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self,
                 observables: Iterable[BaseOperator],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for vals in parameter_sets:
            state = Statevector.from_instruction(self._bind(vals))
            results.append([state.expectation_value(obs) for obs in observables])
        return results

    def evaluate_with_shots(self,
                            observables: Iterable[BaseOperator],
                            parameter_sets: Sequence[Sequence[float]],
                            *,
                            shots: int | None = None,
                            seed: int | None = None) -> List[List[complex]]:
        raw = self.evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy = []
        for row in raw:
            noisy_row = [complex(rng.normal(x.real, max(1e-6, 1/shots)),
                                 rng.normal(x.imag, max(1e-6, 1/shots))) for x in row]
            noisy.append(noisy_row)
        return noisy

    def fidelity_adjacency(self,
                           states: Sequence[BaseOperator],
                           threshold: float,
                           *,
                           secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
            fid = abs((a.dag() * b)[0, 0])**2
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    def random_network(self,
                       qnn_arch: Sequence[int],
                       samples: int) -> tuple[Sequence[int], List[List[Operator]], List[tuple[Operator, Operator]], Operator]:
        target_unitary = self._random_unitary(qnn_arch[-1])
        training_data = [(self._random_state(len(target_unitary.dims[0])),
                          target_unitary * self._random_state(len(target_unitary.dims[0]))) for _ in range(samples)]
        unitaries: List[List[Operator]] = [[]]
        for layer in range(1, len(qnn_arch)):
            layer_ops: List[Operator] = []
            for _ in range(qnn_arch[layer]):
                op = self._random_unitary(qnn_arch[layer-1] + 1)
                if qnn_arch[layer] > 1:
                    op = op.tensor(self._identity(qnn_arch[layer]-1))
                    op = self._swap(op, qnn_arch[layer-1], qnn_arch[layer-1] + _)
                layer_ops.append(op)
            unitaries.append(layer_ops)
        return qnn_arch, unitaries, training_data, target_unitary

    # helper methods
    def _random_unitary(self, dim: int) -> Operator:
        matrix = np.random.normal(size=(dim, dim)) + 1j*np.random.normal(size=(dim, dim))
        q, _ = np.linalg.qr(matrix)
        return Operator(q)

    def _random_state(self, dim: int) -> Operator:
        vec = np.random.normal(size=(dim, 1)) + 1j*np.random.normal(size=(dim, 1))
        vec /= np.linalg.norm(vec)
        return Operator(vec)

    def _identity(self, n: int) -> Operator:
        return Operator(np.eye(2**n))

    def _swap(self, op: Operator, source: int, target: int) -> Operator:
        # placeholder implementation; real swap would permute indices
        return op

__all__ = ["HybridEstimator"]
