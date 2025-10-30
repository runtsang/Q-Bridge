"""Hybrid estimator that evaluates a parametrised quantum circuit.

The class mirrors the API of the classical estimator but operates on a
Qiskit circuit.  It can optionally use a quantum “quanvolution” filter
(`Conv`) that produces expectation values, and it can inject shot noise
via the simulator backend.
"""

import numpy as np
import qiskit
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit import execute, Aer
from collections.abc import Iterable, Sequence
from typing import List, Optional

class Conv:
    """Quantum quanvolution filter with a parametrised rotation per pixel."""
    def __init__(self,
                 kernel_size: int = 2,
                 backend: Optional[qiskit.providers.Backend] = None,
                 shots: int = 100,
                 threshold: float = 127) -> None:
        self.n_qubits = kernel_size ** 2
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold

        self._circuit = QuantumCircuit(self.n_qubits, self.n_qubits)
        self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]

        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += qiskit.circuit.random.random_circuit(self.n_qubits, 2)

        self._circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        """Execute the filter on a 2‑D block of data.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size) with integer pixel values.

        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        flat = np.reshape(data, (self.n_qubits,))

        param_binds = [{self.theta[i]: np.pi if val > self.threshold else 0}
                       for i, val in enumerate(flat)]

        job = execute(self._circuit,
                      backend=self.backend,
                      shots=self.shots,
                      parameter_binds=param_binds)
        result = job.result().get_counts(self._circuit)

        # Compute mean number of |1> outcomes
        total = 0
        for bitstring, count in result.items():
            ones = sum(int(b) for b in bitstring)
            total += ones * count
        return total / (self.shots * self.n_qubits)


class HybridEstimator:
    """Quantum estimator evaluating expectation values for a parametrised circuit.

    Parameters
    ----------
    circuit : QuantumCircuit
        The circuit to evaluate.  Parameters of the circuit are inferred
        from the circuit itself.
    shots : int | None, optional
        If supplied, Gaussian noise with variance 1/shots is added to the
        expectation values to emulate shot statistics.
    seed : int | None, optional
        Random seed for reproducibility of the noise.
    backend : qiskit.providers.Backend | None, optional
        Quantum simulator or hardware backend.  Defaults to the Aer qasm simulator.
    """
    def __init__(self,
                 circuit: QuantumCircuit,
                 *,
                 shots: Optional[int] = None,
                 seed: Optional[int] = None,
                 backend: Optional[qiskit.providers.Backend] = None) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self.shots = shots
        self.seed = seed
        self.backend = backend or Aer.get_backend("qasm_simulator")
        if shots is not None:
            self.rng = np.random.default_rng(seed)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self,
                 observables: Iterable[BaseOperator],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable.

        Parameters
        ----------
        observables : Iterable[BaseOperator]
            Quantum operators whose expectation value is to be evaluated.
        parameter_sets : Sequence[Sequence[float]]
            Iterable of parameter vectors to bind to the circuit.

        Returns
        -------
        List[List[complex]]
            Nested list of expectation values for each parameter set.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            bound = self._bind(values)
            state = Statevector.from_instruction(bound)
            row = [state.expectation_value(obs) for obs in observables]
            if self.shots is not None:
                noise = self.rng.normal(0, 1 / np.sqrt(self.shots), size=len(row))
                row = [float(val + noise[i]) for i, val in enumerate(row)]
            results.append(row)
        return results


__all__ = ["HybridEstimator", "Conv"]
