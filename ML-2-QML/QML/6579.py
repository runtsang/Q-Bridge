"""Fast estimator for parameterised quantum circuits with support for shot‑limited sampling and hybrid observables."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List

import numpy as np
from qiskit import QuantumCircuit, execute
from qiskit.circuit import Parameter
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.providers.aer import AerSimulator

class ConvCircuit:
    """A minimal quantum filter that emulates a quanvolution layer."""
    def __init__(self, kernel_size: int = 2, backend: AerSimulator | None = None,
                 shots: int = 1024, threshold: float = 127.0) -> None:
        self.n_qubits = kernel_size ** 2
        self.backend = backend or AerSimulator()
        self.shots = shots
        self.threshold = threshold
        self._circuit = QuantumCircuit(self.n_qubits)
        self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        """Return the average probability of measuring |1> across all qubits."""
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0.0
            param_binds.append(bind)

        job = execute(self._circuit,
                      self.backend,
                      shots=self.shots,
                      parameter_binds=param_binds)
        result = job.result().get_counts(self._circuit)

        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val

        return counts / (self.shots * self.n_qubits)

def Conv() -> ConvCircuit:
    """Factory returning a quantum convolution filter."""
    return ConvCircuit()

class FastBaseEstimator:
    """Evaluate expectation values of observables for a parameterised circuit."""
    def __init__(self, circuit: QuantumCircuit, backend: AerSimulator | None = None, shots: int = 1024) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self.backend = backend or AerSimulator()
        self.shots = shots

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self,
                 observables: Iterable[BaseOperator],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        """Return a matrix of expectation values for each parameter set.

        Parameters
        ----------
        observables:
            Iterable of quantum operators.
        parameter_sets:
            Sequence of parameter vectors matching the circuit.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            bound_circ = self._bind(values)
            state = Statevector.from_instruction(bound_circ)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        return results

__all__ = ["FastBaseEstimator", "Conv"]
