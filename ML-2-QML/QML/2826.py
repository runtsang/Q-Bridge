import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from collections.abc import Iterable, Sequence
from typing import List

class QuanvCircuit:
    """Variational quanvolution circuit used in hybrid models."""
    def __init__(self, kernel_size: int, threshold: float = 0.5, shots: int = 100):
        self.n_qubits = kernel_size ** 2
        self._circuit = QuantumCircuit(self.n_qubits)
        self.theta = [self._circuit.parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()
        self.backend = Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> float:
        """Execute the circuit on a single 2‑D data patch."""
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
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

class FastBaseEstimator:
    """Evaluate expectation values for a parametrized quantum circuit."""
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
            row = [state.expectation_value(observable) for observable in observables]
            results.append(row)
        return results

class ConvEstimator:
    """
    Hybrid estimator that couples a quanvolution circuit with a
    FastBaseEstimator.  The ``evaluate`` method accepts a list of
    parameter sets (each a flattened 2‑D patch) and a list of
    quantum observables.  It returns the expectation values for
    each observable and parameter set, enabling batched evaluation
    in research workflows.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.5, shots: int = 100):
        self.circuit = QuanvCircuit(kernel_size, threshold, shots)
        self.estimator = FastBaseEstimator(self.circuit._circuit)

    def evaluate(
        self,
        parameter_sets: Sequence[Sequence[float]],
        observables: Iterable[BaseOperator] | None = None,
    ) -> List[List[complex]]:
        if observables is None:
            observables = []
        return self.estimator.evaluate(observables, parameter_sets)

__all__ = ["ConvEstimator"]
