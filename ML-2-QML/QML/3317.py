"""Quantum convolution module with integrated fast expectation‑value estimation."""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.circuit.random import random_circuit
from typing import Iterable, List, Sequence

# --------------------------------------------------------------------------- #
#  Lightweight quantum estimator utilities (from FastBaseEstimator reference)
# --------------------------------------------------------------------------- #
class FastBaseEstimator:
    """Evaluate a parametrized quantum circuit for a batch of parameter sets."""

    def __init__(self, circuit: QuantumCircuit) -> None:
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


class FastEstimator(FastBaseEstimator):
    """Adds optional Gaussian shot noise to the deterministic estimator."""

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in raw:
            noisy_row = [
                complex(
                    rng.normal(float(val.real), max(1e-6, 1 / shots))
                    + 1j
                    * rng.normal(float(val.imag), max(1e-6, 1 / shots))
                )
                for val in row
            ]
            noisy.append(noisy_row)
        return noisy


# --------------------------------------------------------------------------- #
#  ConvHybrid: Quantum convolution with fast expectation evaluation
# --------------------------------------------------------------------------- #
class ConvHybrid:
    """Quantum analogue of the classical ConvHybrid filter."""

    def __init__(self, kernel_size: int = 2, shots: int = 100, threshold: float = 0.5) -> None:
        self.n_qubits = kernel_size ** 2
        self._circuit = QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold

    def run(self, data):
        """Execute the quantum filter on classical data."""
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)
        job = qiskit.execute(
            self._circuit, self.backend, shots=self.shots, parameter_binds=param_binds
        )
        result = job.result().get_counts(self._circuit)
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Evaluate the circuit for multiple parameter sets using FastBaseEstimator."""
        estimator = FastBaseEstimator(self._circuit)
        return estimator.evaluate(observables, parameter_sets)


__all__ = ["ConvHybrid", "FastBaseEstimator", "FastEstimator"]
