from __future__ import annotations

import numpy as np
import qiskit
from qiskit import assemble, transpile
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from typing import Iterable, Sequence, List

class QuantumCircuitWrapper:
    """Wrapper around a parametrised circuit executed on a chosen backend."""
    def __init__(self, n_qubits: int, backend, shots: int = 1024):
        self._circuit = QuantumCircuit(n_qubits)
        all_qubits = list(range(n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probabilities = counts / self.shots
            return np.sum(states * probabilities)

        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])

class HybridFunction:
    """Differentiable interface between NumPy and the quantum circuit."""
    def __init__(self, circuit: QuantumCircuitWrapper, shift: float = np.pi / 2):
        self.circuit = circuit
        self.shift = shift

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return self.circuit.run(inputs)

class Hybrid:
    """Hybrid layer that forwards activations through a quantum circuit."""
    def __init__(self, n_qubits: int, backend, shots: int = 1024, shift: float = np.pi / 2):
        self.quantum_circuit = QuantumCircuitWrapper(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return HybridFunction(self.quantum_circuit, self.shift).forward(inputs)

class FastHybridEstimator:
    """Hybrid estimator that evaluates a parametrised quantum circuit with optional shot noise."""
    def __init__(self,
                 circuit: QuantumCircuit,
                 backend,
                 shots: int = 1024,
                 shift: float = np.pi / 2) -> None:
        self.circuit = circuit
        self.backend = backend
        self.shots = shots
        self.shift = shift
        self.quantum = QuantumCircuitWrapper(circuit.num_qubits, backend, shots)

    def _bind_circuit(self, parameter_values: Sequence[float]):
        if len(parameter_values)!= len(self.circuit.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.circuit.parameters, parameter_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self,
                 observables: Iterable[BaseOperator],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            bound = self._bind_circuit(values)
            state = Statevector.from_instruction(bound)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

__all__ = ["FastHybridEstimator"]
