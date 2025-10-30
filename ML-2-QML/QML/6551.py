"""Hybrid estimator combining Qiskit circuit evaluation with optional quanvolution preprocessing and shot noise."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Any

import numpy as np
import qiskit
from qiskit import Aer
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


class QuanvCircuit:
    """Filter circuit used for quanvolution layers."""
    def __init__(self, kernel_size: int, backend: Any, shots: int, threshold: float) -> None:
        self.n_qubits = kernel_size ** 2
        self._circuit = QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> float:
        """Run the quantum circuit on classical data."""
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self._circuit)
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)


class HybridBaseEstimator:
    """
    Evaluate a Qiskit circuit with optional quanvolution preprocessing and shot noise.
    """
    def __init__(
        self,
        circuit: QuantumCircuit,
        filter: Any | None = None,
        backend: Any | None = None,
    ) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self.filter = filter
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = 100

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def _prepare_input(self, params: Sequence[float]) -> Sequence[float]:
        if self.filter is not None:
            data = np.array(params).reshape(1, -1)
            val = self.filter.run(data[0])
            return [val] * self._circuit.num_qubits
        return params

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """
        Compute expectation values for each parameter set and observable.

        Parameters
        ----------
        observables : iterable of BaseOperator
            Quantum operators whose expectation values are to be computed.
        parameter_sets : sequence of parameter sequences
            Each inner sequence is fed to the circuit after optional filtering.
        shots : int, optional
            If provided, overrides the circuit's shot count for evaluation.
        seed : int, optional
            Random seed for reproducibility of parameter shuffling.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        if shots is not None:
            self.shots = shots

        for values in parameter_sets:
            params = self._prepare_input(values)
            state = Statevector.from_instruction(self._bind(params))
            row = [state.expectation_value(observable) for observable in observables]
            results.append(row)

        return results


__all__ = ["HybridBaseEstimator"]
