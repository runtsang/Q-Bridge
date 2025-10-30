import itertools
from typing import Iterable, Sequence, List
import numpy as np
import networkx as nx
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import Sampler as StatevectorSampler
from qiskit.quantum_info import Statevector, Operator
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN

class SamplerQNN:
    """Quantum sampler that mirrors the classical SamplerQNN interface."""
    def __init__(self) -> None:
        self.inputs = ParameterVector("input", 2)
        self.weights = ParameterVector("weight", 4)
        qc = QuantumCircuit(2)
        qc.ry(self.inputs[0], 0)
        qc.ry(self.inputs[1], 1)
        qc.cx(0, 1)
        qc.ry(self.weights[0], 0)
        qc.ry(self.weights[1], 1)
        qc.cx(0, 1)
        qc.ry(self.weights[2], 0)
        qc.ry(self.weights[3], 1)
        self.circuit = qc
        sampler = StatevectorSampler()
        self.sampler_qnn = QSamplerQNN(
            circuit=qc,
            input_params=self.inputs,
            weight_params=self.weights,
            sampler=sampler,
        )

    def sample(self, input_vals: Sequence[float], weight_vals: Sequence[float], shots: int = 1024) -> np.ndarray:
        param_dict = {p: v for p, v in zip(self.inputs, input_vals)} | {p: v for p, v in zip(self.weights, weight_vals)}
        return self.sampler_qnn.sample(param_dict, shots=shots)

class QuantumKernel:
    """Quantum kernel that evaluates overlap of two encoded states."""
    def __init__(self) -> None:
        self.encoder = self._build_encoder()

    def _build_encoder(self) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        x = ParameterVector("x", 2)
        qc.ry(x[0], 0)
        qc.ry(x[1], 1)
        qc.cx(0, 1)
        return qc

    def encode_state(self, params: Sequence[float]) -> Statevector:
        qc = self._build_encoder()
        bound = qc.assign_parameters({p: v for p, v in zip(qc.parameters, params)}, inplace=False)
        return Statevector.from_instruction(bound)

    def kernel_matrix(self, a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> np.ndarray:
        """Compute the Gram matrix via state overlap."""
        kernel = np.zeros((len(a), len(b)), dtype=float)
        for i, pa in enumerate(a):
            sa = self.encode_state(pa)
            for j, pb in enumerate(b):
                sb = self.encode_state(pb)
                kernel[i, j] = abs((sa.dag() * sb)[0, 0]) ** 2
        return kernel

class FastBaseEstimator:
    """Expectation value evaluator for a parametrised quantum circuit."""
    def __init__(self, circuit: QuantumCircuit) -> None:
        self.circuit = circuit
        self.parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.parameters, parameter_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[Operator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        results: List[List[complex]] = []
        for values in parameter_sets:
            bound = self._bind(values)
            state = Statevector.from_instruction(bound)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

def fidelity_adjacency(
    states: Sequence[Statevector],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a weighted graph from pairwise state fidelities of quantum states."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, s1), (j, s2) in itertools.combinations(enumerate(states), 2):
        fid = abs((s1.dag() * s2)[0, 0]) ** 2
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

__all__ = [
    "SamplerQNN",
    "QuantumKernel",
    "FastBaseEstimator",
    "fidelity_adjacency",
]
