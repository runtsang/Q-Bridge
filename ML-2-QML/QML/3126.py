"""Quantum convolutional filter with variational parameters.

The quantum implementation mirrors the classical module’s API while
providing a variational circuit that can be trained on a quantum
device.  It uses Qiskit’s Aer simulator by default but can be
configured to run on any supported backend.

Key design choices
------------------
- Each 2×2 image patch maps to a 4‑qubit register.
- A parameterised RX rotation encodes the pixel value relative to a
  threshold.
- A random two‑qubit entangling layer introduces expressive
  correlations.
- Measurement in the Z basis yields the probability of |1⟩.
- The mean probability across qubits approximates the classical
  activation.

Usage
-----
>>> from Conv__gen051 import ConvGen051
>>> qc = ConvGen051(kernel_size=2, backend="qasm_simulator",
                    shots=500, threshold=0.5)
>>> data = np.random.randint(0, 2, (2,2))
>>> qc.run(data)
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit
from qiskit import execute

class ConvGen051:
    """
    Variational quantum filter for 2×2 image patches.
    """
    def __init__(self,
                 kernel_size: int = 2,
                 backend: str | qiskit.providers.Provider = "qasm_simulator",
                 shots: int = 200,
                 threshold: float = 0.5) -> None:
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.n_qubits = kernel_size ** 2

        # Build the parametric circuit
        self.circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        self.circuit += random_circuit(self.n_qubits, 2)
        self.circuit.measure_all()

        # Resolve backend
        if isinstance(backend, str):
            self.backend = qiskit.Aer.get_backend(backend)
        else:
            self.backend = backend
        self.shots = shots

    def run(self, data: np.ndarray) -> float:
        """
        Execute the quantum filter on a 2×2 patch.

        Parameters
        ----------
        data : np.ndarray
            2D array of shape (kernel_size, kernel_size) with pixel values
            in [0, 1].  Values above ``threshold`` are encoded as π
            rotations, otherwise 0.

        Returns
        -------
        float
            Average probability of measuring |1⟩ across all qubits.
        """
        flat = data.flatten()
        param_binds = []
        for val in flat:
            bind = {self.theta[i]: np.pi if val > self.threshold else 0.0
                    for i, val in enumerate(flat)}
            param_binds.append(bind)

        job = execute(self.circuit,
                      backend=self.backend,
                      shots=self.shots,
                      parameter_binds=param_binds)
        result = job.result()
        counts_dict = result.get_counts(self.circuit)

        total_ones = 0
        for bitstring, count in counts_dict.items():
            total_ones += count * sum(int(b) for b in bitstring)
        return total_ones / (self.shots * self.n_qubits)

__all__ = ["ConvGen051"]
