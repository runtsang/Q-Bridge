"""Quantum estimator with optional quantum convolution filter and shot‑noise emulation.

The implementation builds on the original FastBaseEstimator by:
* Adding a QuanvCircuit that preprocesses classical data with a parameterised
  RX‑rotation circuit followed by a random two‑layer circuit.
* Allowing the filter output to be treated as an additional observable.
* Adding Gaussian noise to expectation values when a shot count is requested.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
import qiskit
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.providers.aer import AerSimulator


class QuanvCircuit:
    """Quantum convolution filter that maps a 2‑D data patch to a scalar.

    The circuit prepares a register of ``n_qubits`` qubits, applies an
    RX rotation conditioned on the classical data, runs a random two‑layer
    entangling circuit, measures all qubits, and returns the average
    probability of measuring |1>.
    """

    def __init__(self, kernel_size: int = 2, threshold: float = 127, shots: int = 100) -> None:
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.backend = AerSimulator()
        self.circuit = QuantumCircuit(self.n_qubits, self.n_qubits)
        self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        self.circuit += qiskit.circuit.random.random_circuit(self.n_qubits, depth=2)
        self.circuit.measure(range(self.n_qubits), range(self.n_qubits))

    def run(self, data: np.ndarray) -> float:
        """Execute the filter on a 2‑D array and return the average |1> probability."""
        data_flat = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data_flat:
            bind = {self.theta[i]: np.pi if val > self.threshold else 0 for i, val in enumerate(dat)}
            param_binds.append(bind)
        job = self.backend.run(
            self.circuit, shots=self.shots, parameter_binds=param_binds
        )
        result = job.result()
        counts = result.get_counts(self.circuit)
        total_ones = sum(sum(int(bit) for bit in key) * val for key, val in counts.items())
        return total_ones / (self.shots * self.n_qubits)


class FastBaseEstimator:
    """Quantum estimator that evaluates a parameterised circuit with optional filtering.

    Parameters
    ----------
    circuit : QuantumCircuit
        The primary parameterised circuit to evaluate.
    filter : QuanvCircuit | None, optional
        Optional quantum convolution filter applied to each data point before
        evaluation. The filter output is appended as an extra observable.
    """

    def __init__(self, circuit: QuantumCircuit, filter: Optional[QuanvCircuit] = None) -> None:
        self.circuit = circuit
        self.filter = filter
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable.

        If ``shots`` is provided, Gaussian noise is added to the deterministic
        expectation values to emulate finite‑shot effects.
        The optional ``filter`` is applied to each data point and its output
        is appended as the last element of each result row.
        """
        observables = list(observables)
        results: List[List[complex]] = []

        rng = np.random.default_rng(seed) if shots is not None else None

        for values in parameter_sets:
            # Optional filtering
            filter_val: Optional[float] = None
            if self.filter is not None:
                k = int(np.sqrt(self.filter.n_qubits))
                data_2d = np.array(values).reshape(k, k)
                filter_val = self.filter.run(data_2d)

            bound_circuit = self._bind(values)
            state = Statevector.from_instruction(bound_circuit)
            row = [state.expectation_value(obs) for obs in observables]
            if filter_val is not None:
                row.append(complex(filter_val))

            # Add shot‑noise if requested
            if shots is not None and rng is not None:
                noisy_row = [complex(rng.normal(float(val.real), max(1e-6, 1 / shots))) for val in row]
                results.append(noisy_row)
            else:
                results.append(row)

        return results


__all__ = ["FastBaseEstimator", "QuanvCircuit"]
