"""GraphQNNGen058: Quantum graph neural network with estimator utilities.

This module implements a variational quantum circuit that mirrors the
classical GraphQNN interface.  It builds a layered network of random
unitaries, propagates quantum states through the network, and
constructs a fidelity‑based adjacency graph.  An accompanying
FastBaseEstimator evaluates expectations of arbitrary observables
on the output states.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple, Callable

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, random_unitary
from qiskit.opflow import PauliSumOp

ScalarObservable = Callable[[Statevector], complex]


def _ensure_batch(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized circuit."""
    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[PauliSumOp],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results


class GraphQNNGen058:
    """Variational quantum graph neural network with fidelity graph utilities."""
    def __init__(self, arch: Sequence[int], seed: int | None = None) -> None:
        self.arch = list(arch)
        rng = np.random.default_rng(seed)
        self.unitaries: List[List[QuantumCircuit]] = [[]]
        for layer in range(1, len(arch)):
            num_inputs = arch[layer - 1]
            num_outputs = arch[layer]
            layer_circuits: List[QuantumCircuit] = []
            for _ in range(num_outputs):
                qc = QuantumCircuit(num_inputs + 1)
                qc.append(random_unitary(num_inputs + 1, seed=rng.integers(0, 1 << 30)), range(num_inputs + 1))
                if num_outputs > 1:
                    # Swap new output qubit to the end
                    perm = list(range(num_inputs)) + [num_inputs] + list(range(num_inputs, num_inputs))
                    qc.barrier()
                    qc = qc.compose(qc, front=False, control_qubits=perm)
                layer_circuits.append(qc)
            self.unitaries.append(layer_circuits)

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int, seed: int | None = None):
        """Generate a random quantum network and training data."""
        rng = np.random.default_rng(seed)
        target_unitary = random_unitary(qnn_arch[-1], seed=rng.integers(0, 1 << 30))
        training_data = []
        for _ in range(samples):
            state = Statevector.random(num_qubits=qnn_arch[-1], seed=rng.integers(0, 1 << 30))
            training_data.append((state, target_unitary @ state))
        return list(qnn_arch), target_unitary, training_data

    def _partial_trace(self, state: Statevector, keep: Sequence[int]) -> Statevector:
        return state.reduce(keep)

    def feedforward(self, samples: Iterable[Tuple[Statevector, Statevector]]) -> List[List[Statevector]]:
        """Propagate each sample state through the network and return layer‑wise states."""
        stored: List[List[Statevector]] = []
        for sample, _ in samples:
            layerwise = [sample]
            current = sample
            for layer in range(1, len(self.arch)):
                qc = QuantumCircuit(self.arch[layer])
                for gate in self.unitaries[layer]:
                    qc.append(gate, range(self.arch[layer]))
                state = current.evolve(qc)
                current = self._partial_trace(state, list(range(self.arch[layer])))
                layerwise.append(current)
            stored.append(layerwise)
        return stored

    @staticmethod
    def state_fidelity(a: Statevector, b: Statevector) -> float:
        """Return the absolute squared overlap between pure states."""
        return abs((a.dag() @ b)[0, 0]) ** 2

    @staticmethod
    def fidelity_adjacency(
        states: Sequence[Statevector],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNNGen058.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    def estimator(self, observables: Iterable[PauliSumOp]) -> FastBaseEstimator:
        """Return a FastBaseEstimator that evaluates the circuit for given parameter sets."""
        # Build a flat circuit that concatenates all layer unitaries
        circuit = QuantumCircuit(self.arch[-1])
        for layer in range(1, len(self.arch)):
            for gate in self.unitaries[layer]:
                circuit.append(gate, range(self.arch[layer]))
        return FastBaseEstimator(circuit)

__all__ = [
    "GraphQNNGen058",
    "FastBaseEstimator",
]
