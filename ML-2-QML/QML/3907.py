"""Quantum convolutional filter with FastEstimator API.

Implements a parameterized circuit over a kernel-sized grid of qubits.
Supports expectation evaluation, shot noise simulation, and dynamic parameter binding.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from collections.abc import Iterable, Sequence
from typing import List


class ConvGen219:
    """Quantum convolutional filter with FastEstimator interface.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the quantum filter kernel (grid of qubits).
    threshold : float, default 127
        Threshold for encoding classical data into rotation angles.
    shots : int, optional
        Number of shots for circuit execution; defaults to 100.
    """

    def __init__(self, kernel_size: int = 2, threshold: float = 127, shots: int | None = None) -> None:
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.n_qubits = kernel_size ** 2
        self.backend = Aer.get_backend("qasm_simulator")
        self.shots = shots or 100
        self._circuit = QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        # Initial Rx rotations parameterized by theta
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        # Add a random 2-layer circuit for entanglement
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        """Execute the circuit once with data encoded via rotation angles.

        Parameters
        ----------
        data : array-like
            2D array of shape (kernel_size, kernel_size) containing classical values.

        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        data = np.reshape(data, (self.n_qubits,))
        bind = {self.theta[i]: np.pi if val > self.threshold else 0 for i, val in enumerate(data)}
        job = execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[bind],
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
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable.

        Parameters are bound to the circuit's theta rotation angles.
        """
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            bound_circ = self._bind_parameters(values)
            state = Statevector.from_instruction(bound_circ)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        if shots is None:
            return results
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in results:
            noisy_row = [
                complex(
                    rng.normal(val.real, max(1e-6, 1 / shots)),
                    rng.normal(val.imag, max(1e-6, 1 / shots)),
                )
                for val in row
            ]
            noisy.append(noisy_row)
        return noisy

    def _bind_parameters(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        """Return a new circuit with the given parameter values bound."""
        if len(parameter_values)!= self.n_qubits:
            raise ValueError("Parameter count mismatch for ConvGen219.")
        mapping = dict(zip(self.theta, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)


__all__ = ["ConvGen219"]
