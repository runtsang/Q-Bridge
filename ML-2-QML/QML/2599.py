"""Hybrid quantum estimator combining expectation evaluation with a
quantum self‑attention block.  The class extends the lightweight
``FastBaseEstimator`` implementation from the original anchor and adds a
``QuantumSelfAttention`` circuit that can be invoked before the main
circuit evaluation.  The public API mirrors the classical version, so
pipelines can switch between the two backends seamlessly.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

# Base estimator copied from the anchor with slight refactor
class _QuantumBaseEstimator:
    """Base class for evaluating parameterised Qiskit circuits."""

    def __init__(self, circuit: QuantumCircuit) -> None:
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, values: Sequence[float]) -> QuantumCircuit:
        if len(values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []
        for vals in parameter_sets:
            state = Statevector.from_instruction(self._bind(vals))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

# Quantum self‑attention helper
class _QuantumSelfAttention:
    """Parameterised self‑attention block implemented with Qiskit."""

    def __init__(self, n_qubits: int = 4) -> None:
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(
        self,
        backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> dict:
        """Execute the attention circuit and return measurement counts."""
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(circuit, backend, shots=shots)
        return job.result().get_counts(circuit)

def _build_quantum_attention(n_qubits: int = 4) -> _QuantumSelfAttention:
    """Factory returning a quantum attention instance."""
    return _QuantumSelfAttention(n_qubits=n_qubits)

class HybridEstimator(_QuantumBaseEstimator):
    """Hybrid quantum estimator that can optionally prepend a quantum
    self‑attention block before evaluating the main circuit.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        attention: Optional[_QuantumSelfAttention] = None,
        backend: Optional[qiskit.providers.Backend] = None,
    ) -> None:
        super().__init__(circuit)
        self.attention = attention or _build_quantum_attention()
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")

    def apply_attention(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> dict:
        """Run the internal quantum attention circuit."""
        return self.attention.run(self.backend, rotation_params, entangle_params, shots=shots)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        use_attention: bool = False,
    ) -> List[List[complex]]:
        """Evaluate the circuit, optionally applying quantum attention first."""
        if use_attention:
            # Assume each parameter set contains rotation, entangle, and main params
            new_params: List[List[float]] = []
            for params in parameter_sets:
                rot = params[:8]
                ent = params[8:16]
                main = params[16:]
                counts = self.apply_attention(rot, ent, shots=1)
                freq = [counts.get('1', 0) / 1.0]
                new_params.append(freq + main)
            parameter_sets = new_params
        return super().evaluate(observables, parameter_sets)

__all__ = ["HybridEstimator"]
