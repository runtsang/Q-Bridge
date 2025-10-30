"""Hybrid estimator combining a Qiskit circuit with optional quantum self‑attention."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Optional

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

# ----------------------------------------------------------------------
# Quantum self‑attention primitive
# ----------------------------------------------------------------------
class QuantumSelfAttention:
    """A minimal self‑attention style block built with Qiskit."""

    def __init__(self, n_qubits: int) -> None:
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(
        self, rotation_params: np.ndarray, entangle_params: np.ndarray
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
    ) -> dict[str, int]:
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, backend, shots=shots)
        return job.result().get_counts(circuit)


# ----------------------------------------------------------------------
# Hybrid estimator
# ----------------------------------------------------------------------
class FastHybridEstimator:
    """Evaluate a Qiskit circuit and optionally prepend a quantum self‑attention block."""

    def __init__(
        self,
        circuit: QuantumCircuit,
        attention: Optional[QuantumSelfAttention] = None,
        *,
        shots: Optional[int] = None,
    ) -> None:
        self.circuit = circuit
        self.attention = attention
        self.shots = shots
        self.backend = qiskit.Aer.get_backend("qasm_simulator") if shots is not None else None

    def _composite_circuit(
        self, rotation_params: np.ndarray, entangle_params: np.ndarray
    ) -> QuantumCircuit:
        """Attach an optional self‑attention circuit before the main circuit."""
        if self.attention is None:
            return self.circuit
        attn_circ = self.attention._build_circuit(rotation_params, entangle_params)
        composite = QuantumCircuit(self.circuit.num_qubits + attn_circ.num_qubits)
        # copy attention circuit
        composite.compose(attn_circ, inplace=True)
        # copy main circuit
        composite.compose(self.circuit, inplace=True, qubits=list(range(self.circuit.num_qubits)))
        return composite

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for params in parameter_sets:
            # split params into rotation and entangle sets for attention
            # assume first 3*n_qubits for rotations, next n_qubits-1 for entangle
            n = self.attention.n_qubits if self.attention else 0
            rot_params = np.asarray(params[: 3 * n]) if n else np.array([], dtype=float)
            ent_params = np.asarray(params[3 * n : 3 * n + n - 1]) if n else np.array([], dtype=float)
            circ = self._composite_circuit(rot_params, ent_params)
            if self.shots:
                job = execute(circ, self.backend, shots=self.shots)
                counts = job.result().get_counts(circ)
                # naive conversion: treat counts as expectation values
                row = [complex(counts.get(op.name, 0)) for op in observables]
            else:
                state = Statevector.from_instruction(circ)
                row = [state.expectation_value(ob) for ob in observables]
            results.append(row)
        return results


__all__ = ["FastHybridEstimator", "QuantumSelfAttention"]
