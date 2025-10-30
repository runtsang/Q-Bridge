"""Hybrid quantum convolution + classifier.

Builds a data‑encoding circuit followed by a depth‑controlled variational
ansatz and extracts class probabilities from Z measurements.  The
architecture mirrors the classical version: the same depth controls
both the encoding and the classifier layers.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp


class ConvClassifier:
    """
    Quantum analogue of the classical ConvClassifier.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the convolution kernel (determines number of qubits).
    threshold : float, default 127
        Threshold for turning data values into π rotations.
    depth : int, default 2
        Number of variational layers.
    shots : int, default 512
        Number of shots for the simulator.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 127,
        depth: int = 2,
        shots: int = 512,
    ) -> None:
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.n_qubits = kernel_size ** 2
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")

        # Parameter vectors
        self.enc_params = ParameterVector("x", self.n_qubits)
        self.var_params = ParameterVector("theta", self.n_qubits * depth)

        # Build the circuit
        self.circuit = QuantumCircuit(self.n_qubits)

        # Data encoding (RX rotations)
        for q in range(self.n_qubits):
            self.circuit.rx(self.enc_params[q], q)

        # Variational layers
        idx = 0
        for _ in range(depth):
            for q in range(self.n_qubits):
                self.circuit.ry(self.var_params[idx], q)
                idx += 1
            for q in range(self.n_qubits - 1):
                self.circuit.cz(q, q + 1)

        self.circuit.measure_all()

        # Observables for classification
        self.observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (self.n_qubits - i - 1))
            for i in range(self.n_qubits)
        ]

    def run(self, data: np.ndarray) -> float:
        """
        Execute the circuit on the provided data and return the
        averaged probability of measuring |1> across all qubits.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape ``(kernel_size, kernel_size)``.

        Returns
        -------
        float
            Averaged probability of observing |1>.
        """
        flat = data.reshape(-1)

        # Bind encoding parameters based on the threshold
        bind = {
            self.enc_params[i]: np.pi if val > self.threshold else 0
            for i, val in enumerate(flat)
        }

        job = execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[bind],
        )
        result = job.result().get_counts(self.circuit)

        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val

        return counts / (self.shots * self.n_qubits)


__all__ = ["ConvClassifier"]
