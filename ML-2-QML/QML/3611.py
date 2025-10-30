from __future__ import annotations

import numpy as np
import qiskit
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from collections.abc import Iterable, Sequence
from typing import List


class HybridQuantumFCL:
    """
    Parameterised quantum circuit that emulates a fully‑connected layer.
    Provides `run` for single‑theta execution and `evaluate` for batched
    expectation‑value computation, mirroring the classical FastBaseEstimator
    interface.
    """
    def __init__(self, n_qubits: int = 1, backend: qiskit.providers.Backend | None = None, shots: int = 1024) -> None:
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.theta = Parameter("theta")
        self._circuit = QuantumCircuit(n_qubits)
        self._circuit.h(range(n_qubits))
        self._circuit.barrier()
        self._circuit.ry(self.theta, range(n_qubits))
        self._circuit.measure_all()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit for each theta in `thetas` and return expectation values
        over the computational basis probabilities.
        """
        results = []
        for theta in thetas:
            bound = self._circuit.assign_parameters({self.theta: theta})
            job = qiskit.execute(bound, self.backend, shots=self.shots)
            counts = job.result().get_counts(bound)
            probs = np.array(list(counts.values())) / self.shots
            states = np.array([int(k, 2) for k in counts.keys()], dtype=float)
            expectation = np.sum(states * probs)
            results.append(expectation)
        return np.array(results)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._circuit.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = {param: val for param, val in zip(self._circuit.parameters, parameter_values)}
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self, observables: Iterable[BaseOperator], parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        """
        Batch evaluation of expectation values for each parameter set and observable.
        """
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results


def FCL() -> HybridQuantumFCL:
    """Return a quantum fully‑connected layer with batched evaluation."""
    return HybridQuantumFCL()
