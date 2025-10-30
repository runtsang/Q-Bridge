import itertools
from typing import Iterable, Sequence, List, Tuple
import numpy as np
import networkx as nx
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

class FastBaseEstimator:
    """Minimal estimator for parameterized quantum circuits."""
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

class GraphQNNGen(FastBaseEstimator):
    """Quantum graph‑based neural network with estimator interface."""
    def __init__(self, circuit: QuantumCircuit):
        super().__init__(circuit)

    @staticmethod
    def _random_qubit_unitary(num_qubits: int) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        for q in range(num_qubits):
            qc.rx(np.random.normal(), q)
            qc.ry(np.random.normal(), q)
            qc.rz(np.random.normal(), q)
        return qc

    @staticmethod
    def _random_qubit_state(num_qubits: int) -> Statevector:
        dim = 2 ** num_qubits
        vec = np.random.normal(size=(dim,)) + 1j * np.random.normal(size=(dim,))
        vec /= np.linalg.norm(vec)
        return Statevector(vec)

    @staticmethod
    def random_training_data(unitary: QuantumCircuit, samples: int) -> List[Tuple[Statevector, Statevector]]:
        dataset: List[Tuple[Statevector, Statevector]] = []
        num_qubits = unitary.num_qubits
        for _ in range(samples):
            state = GraphQNNGen._random_qubit_state(num_qubits)
            target_state = state.evolve(unitary)
            dataset.append((state, target_state))
        return dataset

    @staticmethod
    def random_network(arch: List[int], samples: int):
        target_circuit = GraphQNNGen._random_qubit_unitary(arch[-1])
        training_data = GraphQNNGen.random_training_data(target_circuit, samples)
        circuits: List[List[QuantumCircuit]] = [[]]
        for layer in range(1, len(arch)):
            num_inputs = arch[layer - 1]
            num_outputs = arch[layer]
            layer_ops: List[QuantumCircuit] = []
            for _ in range(num_outputs):
                op = GraphQNNGen._random_qubit_unitary(num_inputs + 1)
                layer_ops.append(op)
            circuits.append(layer_ops)
        return arch, circuits, training_data, target_circuit

    def _partial_trace(self, state: Statevector, keep: Sequence[int]) -> Statevector:
        trace_out = [i for i in range(state.num_qubits) if i not in keep]
        return state.partial_trace(trace_out)

    def _layer_channel(self, arch: Sequence[int], circuits: Sequence[Sequence[QuantumCircuit]],
                       layer: int, input_state: Statevector) -> Statevector:
        num_inputs = arch[layer - 1]
        num_outputs = arch[layer]
        zero_state = Statevector.from_label('0' * num_outputs)
        composite = input_state.tensor(zero_state)
        for gate in circuits[layer]:
            composite = composite.evolve(gate)
        return self._partial_trace(composite, range(num_inputs))

    def feedforward(self, arch: Sequence[int], circuits: Sequence[Sequence[QuantumCircuit]],
                    samples: Iterable[Tuple[Statevector, Statevector]]) -> List[List[Statevector]]:
        stored_states: List[List[Statevector]] = []
        for sample, _ in samples:
            layerwise = [sample]
            current_state = sample
            for layer in range(1, len(arch)):
                current_state = self._layer_channel(arch, circuits, layer, current_state)
                layerwise.append(current_state)
            stored_states.append(layerwise)
        return stored_states

    @staticmethod
    def state_fidelity(a: Statevector, b: Statevector) -> float:
        return abs(np.vdot(a.data, b.data)) ** 2

    @staticmethod
    def fidelity_adjacency(states: Sequence[Statevector], threshold: float,
                           *, secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNNGen.state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    def evaluate(self,
                 observables: Iterable[BaseOperator],
                 parameter_sets: Sequence[Sequence[float]],
                 *,
                 shots: int | None = None,
                 seed: int | None = None) -> List[List[complex]]:
        base = super().evaluate(observables, parameter_sets)
        if shots is None:
            return base
        rng = np.random.default_rng(seed)
        noisy = []
        for row in base:
            noisy_row = [rng.normal(val, max(1e-6, 1 / shots)) for val in row]
            noisy.append(noisy_row)
        return noisy

__all__ = [
    "GraphQNNGen",
    "FastBaseEstimator",
]
