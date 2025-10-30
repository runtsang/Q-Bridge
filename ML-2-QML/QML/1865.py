"""
ConvEnhanced: Quantum convolutional filter.

This module implements ConvEnhanced as a small wrapper around a
parameterised Qiskit circuit that mimics a convolutional filter.
The circuit:
- Uses a grid of qubits equal to the kernel size squared.
- Applies RX rotations with data‑dependent angles
  (π if the pixel value exceeds a threshold, 0 otherwise).
- Adds a fixed entangling layer.
- Measures all qubits and returns the average probability of
  observing |1⟩.

The class is fully drop‑in compatible with the classical
implementation and can be used in hybrid training pipelines.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit import Aer, execute

class ConvEnhanced:
    """
    Args:
        kernel_size (int): Size of the convolution kernel.
        threshold (float): Threshold used to map pixel values to
            rotation angles.  Values > threshold are mapped to π.
        backend (qiskit.providers.baseprovider.BaseProvider): Qiskit
            backend used for execution.  Defaults to Aer qasm simulator.
        shots (int): Number of shots for the simulation.
    """

    def __init__(self,
                 kernel_size: int = 2,
                 threshold: float = 127.0,
                 backend=None,
                 shots: int = 1024) -> None:
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.n_qubits = kernel_size ** 2
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self._circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        # Define parameters for each qubit
        self.theta = [Parameter(f"θ{i}") for i in range(self.n_qubits)]
        for i, t in enumerate(self.theta):
            qc.rx(t, i)
        # Simple entangling layer
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
        qc.barrier()
        qc.measure_all()
        return qc

    def run(self, data: np.ndarray) -> float:
        """
        Execute the circuit on data and return the average
        probability of measuring |1⟩ across all qubits.

        Parameters
        ----------
        data : np.ndarray
            2‑D array with shape (kernel_size, kernel_size).
        """
        if data.shape!= (self.kernel_size, self.kernel_size):
            raise ValueError(
                f"Data shape must be {(self.kernel_size, self.kernel_size)}"
            )
        # Flatten and binarise data relative to threshold
        flat = data.flatten()
        angles = np.pi * (flat > self.threshold).astype(float)
        param_dict = {self.theta[i]: angle for i, angle in enumerate(angles)}
        # Execute
        job = execute(self._circuit,
                      self.backend,
                      shots=self.shots,
                      parameter_binds=[param_dict])
        result = job.result()
        counts = result.get_counts()
        # Compute average |1> probability
        total = 0
        for bitstring, cnt in counts.items():
            ones = bitstring.count("1")
            total += ones * cnt
        avg_prob = total / (self.shots * self.n_qubits)
        return avg_prob

__all__ = ["ConvEnhanced"]
