"""Hybrid estimator combining quantum circuits and self‑attention."""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from collections.abc import Iterable, Sequence
from typing import List, Sequence as Seq

class QuantumSelfAttention:
    """Self‑attention style block implemented as a parameterised circuit."""
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure_all()
        return circuit

    def run(
        self,
        backend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ):
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = execute(circuit, backend, shots=shots)
        return job.result().get_counts(circuit)


class HybridBaseEstimator:
    """Fast deterministic and noisy evaluation for quantum circuits."""
    def __init__(self, circuit: QuantumCircuit, *, use_attention: bool = False, n_qubits: int | None = None, backend=None):
        self.base_circuit = circuit
        self.use_attention = use_attention
        self.backend = backend or Aer.get_backend("qasm_simulator")
        if use_attention:
            if n_qubits is None:
                raise ValueError("n_qubits must be specified when use_attention is True")
            self.attention = QuantumSelfAttention(n_qubits)
        else:
            self.attention = None

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """Compute expectation values for each parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []
        for params in parameter_sets:
            if self.attention is not None:
                # Expect first 3*n_qubits entries for rotation, next n_qubits-1 for entangle
                rot_len = 3 * self.attention.n_qubits
                ent_len = self.attention.n_qubits - 1
                rotation_params = np.asarray(params[:rot_len])
                entangle_params = np.asarray(params[rot_len:rot_len + ent_len])
                attn_circ = self.attention._build_circuit(rotation_params, entangle_params)
                circuit = attn_circ + self.base_circuit
            else:
                circuit = self.base_circuit
            state = Statevector.from_instruction(circuit)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        if shots is None:
            return results
        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in results:
            noisy_row = [complex(rng.normal(val.real, max(1e-6, 1 / shots)),
                                 rng.normal(val.imag, max(1e-6, 1 / shots))) for val in row]
            noisy.append(noisy_row)
        return noisy


__all__ = ["HybridBaseEstimator", "QuantumSelfAttention"]
