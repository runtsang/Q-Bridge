"""Quantum circuit for fraud detection – a single‑qubit variational circuit.

The circuit is inspired by the simple two‑qubit QCNet head but reduced to a
single qubit for efficiency.  It accepts a 1‑D array of rotation angles and
returns the expectation value of the Z‑observable.  The class is deliberately
simple so that it can be injected into the PyTorch hybrid head defined in
the ML module.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import Aer, assemble, transpile
from qiskit.circuit import Parameter


class FraudDetectorHybrid:
    """
    Quantum circuit that implements a parameterised expectation layer.

    Parameters
    ----------
    n_qubits : int, default 1
        Number of qubits in the circuit.
    shots : int, default 1024
        Number of shots for the simulator.
    """

    def __init__(self, n_qubits: int = 1, shots: int = 1024) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")
        self.theta = Parameter("theta")

        # Build a minimal circuit: H → RY(θ) → measure
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self._circuit.h(range(n_qubits))
        self._circuit.barrier()
        self._circuit.ry(self.theta, range(n_qubits))
        self._circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """Execute the circuit for a batch of rotation angles.

        Parameters
        ----------
        thetas : np.ndarray
            1‑D array of angles, one per sample.

        Returns
        -------
        np.ndarray
            1‑D array of expectation values.
        """
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: float(theta)} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(counts: dict) -> float:
            values = np.array(list(counts.keys()), dtype=float)
            probs = np.array(list(counts.values())) / self.shots
            return np.sum(values * probs)

        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])

__all__ = ["FraudDetectorHybrid"]
