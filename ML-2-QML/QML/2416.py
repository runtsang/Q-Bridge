"""Hybrid fast estimator for Qiskit circuits with optional self‑attention and shot‑noise.

Defines FastBaseEstimator that can evaluate expectation values or counts for a parameterized circuit.
The SelfAttention factory builds a simple attention‑style circuit.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Callable, List, Optional

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.providers import Backend


class QuantumSelfAttention:
    """Quantum circuit implementing a simple self‑attention style block."""

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def build(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> QuantumCircuit:
        """Return a circuit that applies rotations and nearest‑neighbour entanglement."""
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit


def SelfAttention(n_qubits: int) -> QuantumSelfAttention:
    """Convenience factory returning a QuantumSelfAttention instance."""
    return QuantumSelfAttention(n_qubits)


class FastBaseEstimator:
    """Evaluate a parameterised quantum circuit for batches of parameters and observables.

    Parameters
    ----------
    circuit_builder : Callable[[Sequence[float]], QuantumCircuit]
        Function that, given a flat list of parameters, returns a QuantumCircuit.
    backend : Backend, optional
        Qiskit backend to execute the circuit. Defaults to Aer qasm simulator.
    """

    def __init__(
        self,
        circuit_builder: Callable[[Sequence[float]], QuantumCircuit],
        backend: Optional[Backend] = None,
    ) -> None:
        self.circuit_builder = circuit_builder
        self.backend = backend or Aer.get_backend("qasm_simulator")

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
    ) -> List[List[complex | dict]]:
        """
        Evaluate observables for each parameter set.

        If ``shots`` is provided, the circuit is executed on the backend and
        measurement counts are returned.  Otherwise, a Statevector is
        constructed and expectation values are returned.
        """
        results: List[List[complex | dict]] = []
        for params in parameter_sets:
            circuit = self.circuit_builder(params)
            if shots is not None:
                job = qiskit.execute(circuit, self.backend, shots=shots)
                counts = job.result().get_counts(circuit)
                results.append([counts])
            else:
                state = Statevector.from_instruction(circuit)
                row = [state.expectation_value(obs) for obs in observables]
                results.append(row)
        return results


__all__ = ["FastBaseEstimator", "QuantumSelfAttention", "SelfAttention"]
