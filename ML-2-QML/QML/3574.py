"""Quantum convolutional filter (QuanvCircuit) used by the hybrid estimator.

The circuit encodes a 2×2 pixel patch into rotation angles: each pixel value
is thresholded and mapped to either 0 or π.  After a layer of RX gates the
circuit is enriched with a random 2‑depth layer and finally measured in the
computational basis.  The average probability of measuring |1> across all
qubits serves as the quantum feature for the patch.

The implementation is deliberately lightweight and fully compatible with
Aer simulators.  It can be extended to use state‑vector or shot‑based
estimators as needed.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit import Aer, execute


class QuanvCircuit:
    """Quantum convolutional circuit for a 2×2 pixel patch."""

    def __init__(self, kernel_size: int = 2, threshold: float = 127.0) -> None:
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.n_qubits = kernel_size ** 2

        # Build the circuit
        self.circuit = QuantumCircuit(self.n_qubits)
        self.theta = ParameterVector("theta", self.n_qubits)
        for i, th in enumerate(self.theta):
            self.circuit.rx(th, i)
        self.circuit.barrier()
        self.circuit += qiskit.circuit.random.random_circuit(
            self.n_qubits, depth=2, measure=True
        )
        self.circuit.measure_all()

        # Backend for simulation
        self.backend = Aer.get_backend("qasm_simulator")
        self.shots = 200

    def _encode(self, patch: np.ndarray) -> dict:
        """Map pixel values to rotation angles."""
        bind = {}
        for i, val in enumerate(patch.flat):
            bind[self.theta[i]] = np.pi if val > self.threshold else 0.0
        return bind

    def run(self, patch: np.ndarray) -> float:
        """
        Execute the circuit on a single 2×2 patch.

        Parameters
        ----------
        patch : np.ndarray
            2×2 array with pixel intensities.

        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        bind = self._encode(patch)
        job = execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[bind],
        )
        result = job.result()
        counts = result.get_counts(self.circuit)

        # Compute probability of |1> per qubit
        ones_counts = 0
        for bitstring, cnt in counts.items():
            ones_counts += sum(int(b) for b in bitstring) * cnt

        prob = ones_counts / (self.shots * self.n_qubits)
        return prob


__all__ = ["QuanvCircuit"]
