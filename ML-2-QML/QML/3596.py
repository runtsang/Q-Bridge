"""Quantum convolutional filter with FastBaseEstimator compatibility.

This module implements a quanvolution layer that can be used as a drop‑in
replacement for the classical Conv filter.  It inherits the API of the
FastBaseEstimator from the QML reference, allowing evaluation of
expectation values for arbitrary observables and optional shot‑noise.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit import Aer, execute
from typing import Iterable, List, Sequence

class Conv:
    """Quantum convolutional filter emulating a quanvolution layer.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the square filter; the number of qubits equals
        kernel_size**2.
    backend : qiskit.providers.Backend or None, optional
        Execution backend; defaults to Aer qasm_simulator.
    shots : int, default 100
        Number of shots for simulation.
    threshold : float, default 127
        Pixel value threshold for setting rotation angles.
    """

    def __init__(self,
                 kernel_size: int = 2,
                 backend=None,
                 shots: int = 100,
                 threshold: float = 127) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.shots = shots
        self.threshold = threshold
        self.backend = backend or Aer.get_backend('qasm_simulator')

        # Build parameterised circuit
        self._circuit = QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i, theta in enumerate(self.theta):
            self._circuit.rx(theta, i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

    def run(self, data: np.ndarray | List[int]) -> float:
        """Execute the circuit for a single datum and return the mean |1> prob."""
        flat = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for row in flat:
            bind = {theta: (np.pi if val > self.threshold else 0)
                    for theta, val in zip(self.theta, row)}
            param_binds.append(bind)

        job = execute(self._circuit,
                      self.backend,
                      shots=self.shots,
                      parameter_binds=param_binds)
        result = job.result()
        counts = result.get_counts(self._circuit)

        total_ones = sum(freq * sum(int(b) for b in bitstring)
                         for bitstring, freq in counts.items())
        return total_ones / (self.shots * self.n_qubits)

    # FastBaseEstimator compatibility
    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= self.n_qubits:
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = {theta: val for theta, val in zip(self.theta, parameter_values)}
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self,
                 observables: Iterable[BaseOperator],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable."""
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

__all__ = ["Conv"]
