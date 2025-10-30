"""Pure quantum component for fraud detection.

This module implements a minimal Qiskit variational circuit that
returns the Z‑expectation value for a single parameter. It is designed to
be called from the classical hybrid network but can also be used
stand‑alone for experimentation.
"""

import numpy as np
import qiskit
from qiskit import Aer, transpile, assemble


class FraudDetectionHybridNet:
    """Quantum circuit that outputs the Z‑expectation of a single qubit."""

    def __init__(self, shots: int = 1024):
        self.shots = shots
        self.backend = Aer.get_backend("aer_simulator")
        self.circuit = qiskit.QuantumCircuit(1)
        self.theta = self.circuit.add_parameter("theta")
        self.circuit.h(0)
        self.circuit.ry(self.theta, 0)
        self.circuit.measure_all()

    def run(self, params: np.ndarray) -> np.ndarray:
        """Compute expectation values for an array of parameters."""
        expectations = []
        for p in params:
            compiled = transpile(self.circuit, self.backend)
            qobj = assemble(
                compiled,
                shots=self.shots,
                parameter_binds=[{self.theta: float(p)}],
            )
            job = self.backend.run(qobj)
            result = job.result()
            counts = result.get_counts()
            prob0 = counts.get("0", 0) / self.shots
            prob1 = counts.get("1", 0) / self.shots
            expectations.append(prob0 - prob1)
        return np.array(expectations)


__all__ = ["FraudDetectionHybridNet"]
