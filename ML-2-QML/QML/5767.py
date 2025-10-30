"""Quantum hybrid convolution and sampler.

This module implements a variational quantum circuit that mimics the
behavior of the classical HybridConvSampler.  The circuit receives
an image patch encoded as rotation angles, applies a shallow
entangling layer, and then measures the qubits to produce a
probability distribution.  A state‑vector sampler is used to
evaluate the expectation values efficiently.

Author: gpt-oss-20b
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, transpile
from qiskit.circuit import ParameterVector
from qiskit.primitives import Sampler as QuantumSampler
from qiskit.quantum_info import Statevector


def _entangling_layer(qc: QuantumCircuit, n_qubits: int) -> None:
    """Add a simple entangling pattern."""
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    qc.cx(n_qubits - 1, 0)  # wrap‑around for a ring topology


class QuantumHybridConvSampler:
    """
    Variational circuit that performs a 2×2 quantum convolution and
    returns the mean probability of measuring |1> across all qubits.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the convolutional kernel (number of qubits = kernel_size²).
    threshold : float, default 127
        Pixel intensity threshold used to encode data into rotations.
    shots : int, default 1024
        Number of shots for the simulator.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 127.0,
        shots: int = 1024,
    ) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots

        # Parameter vector for input encoding
        self.input_params = ParameterVector("x", self.n_qubits)

        # Build the circuit
        self.circuit = QuantumCircuit(self.n_qubits)
        # Data encoding: RX rotations conditioned on pixel intensity
        for i in range(self.n_qubits):
            self.circuit.rx(self.input_params[i], i)

        # Entangling layer
        _entangling_layer(self.circuit, self.n_qubits)

        # Measurement
        self.circuit.measure_all()

        # Backend and sampler
        self.backend = Aer.get_backend("aer_simulator")
        self.sampler = QuantumSampler(backend=self.backend)

    def run(self, patch: np.ndarray) -> float:
        """
        Execute the circuit on a single image patch.

        Parameters
        ----------
        patch : np.ndarray
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Mean probability of measuring |1> across all qubits.
        """
        if patch.shape!= (self.kernel_size, self.kernel_size):
            raise ValueError(f"Expected patch shape {(self.kernel_size, self.kernel_size)}, got {patch.shape}")

        # Flatten and encode pixel values into rotation angles
        flat = patch.flatten()
        param_binds = {
            self.input_params[i]: np.pi if val > self.threshold else 0.0
            for i, val in enumerate(flat)
        }

        # Execute with the sampler
        counts = self.sampler.run(
            self.circuit,
            parameter_binds=[param_binds],
            shots=self.shots,
        ).result().get_counts(self.circuit)

        # Compute mean probability of |1> across all qubits
        total_ones = 0
        total_counts = 0
        for bitstring, freq in counts.items():
            ones = bitstring.count("1")
            total_ones += ones * freq
            total_counts += freq

        mean_prob = total_ones / (total_counts * self.n_qubits)
        return mean_prob

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(kernel_size={self.kernel_size}, "
            f"threshold={self.threshold}, shots={self.shots})"
        )


__all__ = ["QuantumHybridConvSampler"]
