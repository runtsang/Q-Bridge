"""Hybrid estimator for quantum circuits with optional quantum convolutional filter.

This module defines FastBaseEstimator that evaluates expectation values of
quantum observables for a parametrized circuit.  It extends the original
lightweight implementation by adding:

* Optional quantum convolutional preprocessing (QuanvCircuit) that can be
  composed with the main circuit.
* Automatic backend selection with a default Aer simulator.
* Support for both state‑vector and shot‑based evaluation.
* Flexible observable interface and error handling.

The estimator can be used as a drop‑in replacement for the original
FastBaseEstimator while providing richer quantum‑classical coupling.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
import qiskit
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.circuit.random import random_circuit

# --------------------------------------------------------------------------- #
# Quantum convolutional filter
# --------------------------------------------------------------------------- #
def QuanvCircuit(
    kernel_size: int = 2,
    backend: Optional[qiskit.providers.backend.Backend] = None,
    shots: int = 100,
    threshold: float = 0.0,
) -> QuantumCircuit:
    """Return a quantum circuit that acts as a filter for 2‑D data.

    The implementation is a refactor of the original ``Conv`` seed.  It
    builds a small circuit that rotates each qubit according to the
    corresponding pixel value and then applies a random two‑qubit
    entangling layer.  The circuit is measured in the computational
    basis and the returned expectation is the average number of ``|1>``
    outcomes.

    Args:
        kernel_size: Size of the square kernel.
        backend: Target backend; defaults to Aer qasm simulator.
        shots: Number of shots for the measurement.
        threshold: Pixel threshold that decides whether a rotation of
            ``π`` or ``0`` is applied.

    Returns:
        QuantumCircuit: The filter circuit ready for binding.
    """
    n_qubits = kernel_size ** 2
    circ = QuantumCircuit(n_qubits)
    thetas = [qiskit.circuit.Parameter(f"theta{i}") for i in range(n_qubits)]
    for i in range(n_qubits):
        circ.rx(thetas[i], i)
    circ.barrier()
    circ += random_circuit(n_qubits, 2, seed=42)
    circ.measure_all()

    circ.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
    circ.shots = shots
    circ.threshold = threshold
    circ.n_qubits = n_qubits
    circ.thetas = thetas
    return circ


# --------------------------------------------------------------------------- #
# Main estimator
# --------------------------------------------------------------------------- #
class FastBaseEstimator:
    """Evaluate expectation values for a parametrized quantum circuit."""

    def __init__(
        self,
        circuit: QuantumCircuit,
        *,
        filter_circuit: Optional[QuantumCircuit] = None,
        backend: Optional[qiskit.providers.backend.Backend] = None,
    ) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)
        self.filter_circuit = filter_circuit
        self.backend = backend or getattr(circuit, "backend", qiskit.Aer.get_backend("qasm_simulator"))

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def _evaluate_filter(self, data: Sequence[float]) -> List[float]:
        if self.filter_circuit is None:
            return list(data)
        bind = {theta: np.pi if val > self.filter_circuit.threshold else 0 for theta, val in zip(self.filter_circuit.thetas, data)}
        circ = self.filter_circuit.assign_parameters(bind, inplace=False)
        job = qiskit.execute(circ, self.filter_circuit.backend, shots=self.filter_circuit.shots)
        result = job.result()
        counts = result.get_counts(circ)
        total = 0
        for key, val in counts.items():
            total += sum(int(bit) for bit in key) * val
        avg = total / (self.filter_circuit.shots * self.filter_circuit.n_qubits)
        return [avg]

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            if self.filter_circuit is not None:
                values = self._evaluate_filter(values)
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results


__all__ = ["FastBaseEstimator", "QuanvCircuit"]
