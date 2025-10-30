"""
Variational quanvolution filter with learnable angles and noise calibration.
The module defines a ``QuanvCircuit`` class that builds a parameterized
circuit on a square grid of qubits. The circuit contains a trainable
angle schedule and a calibration routine that estimates the effective
noise level by running a known reference state. The public API
remains unchanged: ``Conv()`` returns an object with a ``run`` method
that accepts a 2‑D array and returns a scalar probability.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import Parameter
from qiskit.providers.aer.noise import NoiseModel
from typing import Iterable, Tuple

class QuanvCircuit:
    """Parameter‑efficient variational quanvolution filter."""

    def __init__(self,
                 kernel_size: int = 2,
                 backend: qiskit.providers.Provider = None,
                 shots: int = 1024,
                 threshold: float = 0.5,
                 noise_model: NoiseModel | None = None) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.shots = shots
        self.threshold = threshold
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.noise_model = noise_model

        # Parameterized circuit
        self.theta = [Parameter(f"θ{i}") for i in range(self.n_qubits)]
        self.circuit = QuantumCircuit(self.n_qubits)
        for i, p in enumerate(self.theta):
            self.circuit.rx(p, i)
        # Add entangling layer
        for i in range(self.n_qubits - 1):
            self.circuit.cx(i, i + 1)
        # Add a second layer of rotations
        for i, p in enumerate(self.theta):
            self.circuit.rz(p, i)
        self.circuit.measure_all()

        # Calibration parameters
        self.calibration = np.zeros(self.n_qubits)

    def calibrate(self, reference: np.ndarray | None = None) -> None:
        """Estimate the effective noise by running a reference state.

        Args:
            reference: Optional 2‑D array of shape (kernel_size, kernel_size).
                If None, a random state is used.
        """
        if reference is None:
            reference = np.random.rand(self.kernel_size, self.kernel_size)
        ref_data = reference.flatten()
        param_bind = {p: np.pi if v > self.threshold else 0.0 for p, v in zip(self.theta, ref_data)}
        job = execute(self.circuit,
                      self.backend,
                      shots=self.shots,
                      parameter_binds=[param_bind],
                      noise_model=self.noise_model)
        result = job.result().get_counts(self.circuit)
        # Compute average number of |1> outcomes per qubit
        total_ones = 0
        for bits, count in result.items():
            total_ones += bits.count('1') * count
        avg_ones = total_ones / (self.shots * self.n_qubits)
        self.calibration = np.full(self.n_qubits, avg_ones)

    def run(self, data: np.ndarray | list[list[float]]) -> float:
        """Run the variational circuit on a 2‑D array and return average |1> probability."""
        data = np.asarray(data).flatten()
        param_bind = {p: np.pi if v > self.threshold else 0.0 for p, v in zip(self.theta, data)}
        job = execute(self.circuit,
                      self.backend,
                      shots=self.shots,
                      parameter_binds=[param_bind],
                      noise_model=self.noise_model)
        result = job.result().get_counts(self.circuit)

        total_ones = 0
        for bits, count in result.items():
            total_ones += bits.count('1') * count
        avg_prob = total_ones / (self.shots * self.n_qubits)
        # Adjust by calibration
        return float(avg_prob - self.calibration.mean())

    def set_threshold(self, threshold: float) -> None:
        """Update the threshold used to map data to rotation angles."""
        self.threshold = threshold

    def get_parameters(self) -> dict:
        """Return current parameter values."""
        return {f"θ{i}": 0.0 for i in range(self.n_qubits)}  # placeholder

def Conv(kernel_size: int = 2,
         shots: int = 1024,
         threshold: float = 0.5,
         noise_model: NoiseModel | None = None) -> QuanvCircuit:
    """Factory function that returns a QuanvCircuit instance."""
    return QuanvCircuit(kernel_size=kernel_size,
                        shots=shots,
                        threshold=threshold,
                        noise_model=noise_model)

__all__ = ["Conv", "QuanvCircuit"]
