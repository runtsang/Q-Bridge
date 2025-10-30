import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from collections.abc import Iterable, Sequence
from typing import List, Iterable

class FastBaseEstimator:
    """
    Lightweight evaluator that computes expectation values of a list of
    qiskit operators for a parametrised circuit.
    """
    def __init__(self, circuit: qiskit.QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> qiskit.QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self,
                 observables: Iterable[BaseOperator],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

class HybridConvEstimator:
    """
    Quantum implementation of a convolutional filter.  The filter is represented
    by a parameterised circuit that operates on a square patch of classical
    data encoded into rotation angles.  The output is the average probability
    of measuring |1⟩ over all qubits.
    """
    def __init__(self,
                 kernel_size: int = 2,
                 backend: qiskit.providers.Provider | None = None,
                 shots: int = 200,
                 threshold: float = 127.0):
        """
        Parameters
        ----------
        kernel_size : int
            Size of the square filter – determines the number of qubits.
        backend : qiskit.providers.Provider
            Qiskit backend for executing the circuit.  Defaults to the local
            Aer simulator.
        shots : int
            Number of measurement shots per evaluation.
        threshold : float
            Classical threshold used to map input pixel values to rotation angles.
        """
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")

        # Build the parametrised circuit
        self._circuit = qiskit.QuantumCircuit(self.n_qubits, self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i, p in enumerate(self.theta):
            self._circuit.rx(p, i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure(range(self.n_qubits), range(self.n_qubits))

        self.estimator = FastBaseEstimator(self._circuit)

    def run(self, data: np.ndarray) -> float:
        """
        Execute the circuit on a single kernel‑sized patch.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size) with values in [0, 255].

        Returns
        -------
        float
            Average probability of measuring |1⟩ across all qubits.
        """
        if data.shape!= (self.kernel_size, self.kernel_size):
            raise ValueError(f"Expected data shape ({self.kernel_size},{self.kernel_size})")
        flat = data.reshape(1, self.n_qubits)
        param_binds = []
        for row in flat:
            bind = {p: np.pi if val > self.threshold else 0.0 for p, val in zip(self.theta, row)}
            param_binds.append(bind)

        job = qiskit.execute(self._circuit,
                             self.backend,
                             shots=self.shots,
                             parameter_binds=param_binds)
        result = job.result().get_counts(self._circuit)
        # Compute average number of |1⟩ over all shots and qubits
        total_ones = 0
        for bitstring, count in result.items():
            total_ones += count * sum(int(b) for b in bitstring)
        return total_ones / (self.shots * self.n_qubits)

    def evaluate(self,
                 observables: Iterable[BaseOperator],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        """
        Evaluate expectation values of the provided observables for each
        parameter set using the FastBaseEstimator.
        """
        return self.estimator.evaluate(observables, parameter_sets)

__all__ = ["HybridConvEstimator"]
