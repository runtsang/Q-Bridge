"""
Quantum decoder used in the hybrid autoencoder.

This module provides a parameterised variational circuit that implements
a fully‑connected layer in a quantum circuit. Each latent dimension
corresponds to a qubit that undergoes a Ry rotation controlled by a
classical parameter. The expectation value of the Z observable is
returned as the decoded feature.
"""

from __future__ import annotations

import numpy as np
from typing import Iterable

from qiskit import QuantumCircuit, Aer
from qiskit.providers.aer import AerSimulator

class QuantumDecoder:
    """Variational quantum decoder with one qubit per latent dimension."""
    def __init__(self, n_qubits: int, backend: AerSimulator | None = None, shots: int = 1024) -> None:
        self.n_qubits = n_qubits
        self.backend = backend or AerSimulator()
        self.shots = shots
        # Build a simple ansatz: H on all qubits, then Ry(theta_i) on each
        self.circuit = QuantumCircuit(n_qubits)
        self.circuit.h(range(n_qubits))
        self.params = [self.circuit.parameter(f"theta_{i}") for i in range(n_qubits)]
        for i, param in enumerate(self.params):
            self.circuit.ry(param, i)
        self.circuit.measure_all()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Evaluate the circuit for a single latent vector.

        Args:
            thetas: Iterable of length n_qubits containing the parameters
                    for the Ry rotations.

        Returns:
            np.ndarray of shape (n_qubits,) containing the expectation
            values of the Z observable for each qubit.
        """
        param_binds = [{self.params[i]: thetas[i]} for i in range(self.n_qubits)]
        job = self.backend.run(self.circuit, shots=self.shots, parameter_binds=param_binds)
        result = job.result()
        counts = result.get_counts(self.circuit)
        expectation = np.zeros(self.n_qubits)
        for i in range(self.n_qubits):
            exp = 0.0
            for state, cnt in counts.items():
                # state string is in little‑endian order
                bit = int(state[::-1][i])
                prob = cnt / self.shots
                exp += (1 if bit == 0 else -1) * prob
            expectation[i] = exp
        return expectation

__all__ = ["QuantumDecoder"]
