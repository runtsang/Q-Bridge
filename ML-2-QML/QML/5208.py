"""Quantum‑guided hybrid module for ConvGenHybrid.

This module provides the quantum part of the hybrid head and
optionally a quantum‑enhanced attention block. It is designed to
be imported into the Python module above and can be used as a
stand‑alone quantum circuit that mimics the behaviour of the
`HybridHead` class.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import assemble, transpile
from qiskit.providers.aer import AerSimulator

class VariationalCircuit:
    """Simple 2‑qubit RX‑RY‑RZ variational circuit with parameter‑dependent expectation."""
    def __init__(self,
                 n_qubits: int = 2,
                 backend: AerSimulator | None = None,
                 shots: int = 512) -> None:
        self.n_qubits = n_qubits
        self.backend = backend or AerSimulator()
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(self.n_qubits)
        # H on all qubits
        self.circuit.h(range(self.n_qubits))
        # Parameterised rotations
        self.theta_params = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i, theta in enumerate(self.theta_params):
            self.circuit.rx(theta, i)
        # Entanglement (simple CNOT chain)
        for i in range(self.n_qubits - 1):
            self.circuit.cx(i, i + 1)
        self.circuit.measure_all()

    def run(self, angles: np.ndarray) -> float:
        """Execute the circuit for the given rotation angles and return Pauli‑Z expectation."""
        if len(angles)!= self.n_qubits:
            raise ValueError("Angle array length must match number of qubits.")
        bind = {self.theta_params[i]: angles[i] for i in range(self.n_qubits)}
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled,
                        shots=self.shots,
                        parameter_binds=[bind])
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        return self._expectation(result)

    def _expectation(self, counts: dict[str, int]) -> float:
        """Compute expectation of Pauli‑Z from measurement counts."""
        probs = np.array(list(counts.values())) / self.shots
        states = np.array([int(k, 2) for k in counts.keys()])
        signs = (-1) ** states
        return float(np.sum(signs * probs))

class QuantumAttention:
    """Quantum‑enhanced self‑attention block using a simple variational circuit per head."""
    def __init__(self,
                 n_heads: int,
                 n_qubits_per_head: int = 1,
                 backend: AerSimulator | None = None,
                 shots: int = 1024) -> None:
        self.n_heads = n_heads
        self.n_qubits_per_head = n_qubits_per_head
        self.backend = backend or AerSimulator()
        self.shots = shots
        self.circuits = [self._build_circuit(h) for h in range(n_heads)]

    def _build_circuit(self, head_idx: int) -> qiskit.QuantumCircuit:
        qc = qiskit.QuantumCircuit(self.n_qubits_per_head)
        theta = qiskit.circuit.Parameter(f"theta{head_idx}")
        for i in range(self.n_qubits_per_head):
            qc.rx(theta, i)
        qc.measure_all()
        return qc

    def run(self, angles: np.ndarray) -> np.ndarray:
        """Run the attention circuit for each head and return expectation values."""
        # angles shape: (batch, seq_len, n_heads)
        batch, seq_len, _ = angles.shape
        outputs = np.zeros((batch, self.n_heads))
        for h in range(self.n_heads):
            avg_angles = angles[:, :, h].mean(axis=1)
            bind = {self.circuits[h].parameters[0]: avg_angles}
            compiled = transpile(self.circuits[h], self.backend)
            qobj = assemble(compiled, shots=self.shots, parameter_binds=[bind])
            job = self.backend.run(qobj)
            result = job.result().get_counts()
            outputs[:, h] = self._expectation(result)
        return outputs

    def _expectation(self, counts: dict[str, int]) -> float:
        probs = np.array(list(counts.values())) / self.shots
        states = np.array([int(k, 2) for k in counts.keys()])
        signs = (-1) ** states
        return float(np.sum(signs * probs))

__all__ = ["VariationalCircuit", "QuantumAttention"]
