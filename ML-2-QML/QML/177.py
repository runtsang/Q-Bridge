"""Quantum‑only version of ConvHybrid.

This module implements a variational circuit that reproduces the behaviour
of the classical Conv filter.  It can be used as a standalone quantum
layer or imported into a hybrid model that wraps it with a classical
convolution.
"""

from __future__ import annotations

import numpy as np
from qiskit import Aer, execute
from qiskit.circuit import Parameter, QuantumCircuit
from typing import Optional, Tuple

class ConvHybrid:
    """
    Stand‑alone quantum circuit that encodes a 2D patch into a
    parameter‑shift differentiable variational circuit.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        n_layers: int = 2,
        shots: int = 100,
        backend: Optional["qiskit.providers.Backend"] = None,
    ) -> None:
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.n_layers = n_layers
        self.shots = shots
        self.n_qubits = kernel_size ** 2

        # Parameters for each qubit
        self.theta = [Parameter(f"theta_{i}") for i in range(self.n_qubits)]

        # Build a reusable circuit template
        self.circuit_template = self._build_circuit_template()

        # Backend
        self.backend = backend or Aer.get_backend("qasm_simulator")

    def _build_circuit_template(self) -> QuantumCircuit:
        """
        Build a parameterised circuit template without data encoding.
        """
        qc = QuantumCircuit(self.n_qubits)
        # Parameterised rotations
        for i in range(self.n_qubits):
            qc.ry(self.theta[i], i)
        # Entangling layer
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
        # Additional depth
        for _ in range(self.n_layers - 1):
            for i in range(self.n_qubits):
                qc.ry(self.theta[i], i)
            for i in range(self.n_qubits - 1):
                qc.cx(i, i + 1)
        return qc

    def _encode(self, patch: np.ndarray) -> np.ndarray:
        """
        Encode a 2D patch into rotation angles.
        Thresholding: > threshold => pi, else 0.
        """
        flat = patch.flatten()
        return np.where(flat > self.threshold, np.pi, 0.0)

    def run(self, patch: np.ndarray) -> float:
        """
        Evaluate the quantum circuit on a single patch.

        Args:
            patch: 2D numpy array of shape (kernel_size, kernel_size).

        Returns:
            float: expectation value of Pauli‑Z on each qubit averaged.
        """
        angles = self._encode(patch)

        qc = self.circuit_template.copy()
        bind_dict = {self.theta[i]: angles[i] for i in range(self.n_qubits)}
        qc.bind_parameters(bind_dict)
        qc.measure_all()

        job = execute(qc, self.backend, shots=self.shots)
        result = job.result()
        counts = result.get_counts(qc)

        # Expectation of Z: +1 for |0>, -1 for |1>
        exp = 0.0
        for bitstring, count in counts.items():
            bits = bitstring[::-1]  # reverse to match qubit ordering
            z_sum = sum(1 if b == "0" else -1 for b in bits)
            exp += z_sum * count
        exp = exp / (self.shots * self.n_qubits)
        return exp

    def grad(self, patch: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the expectation value w.r.t. the rotation angles
        using the parameter‑shift rule.

        Returns:
            np.ndarray of shape (n_qubits,)
        """
        shift = np.pi / 2
        angles = self._encode(patch)

        grads = np.zeros(self.n_qubits, dtype=np.float32)

        for i in range(self.n_qubits):
            # Shift up
            up = angles.copy()
            up[i] += shift
            qc_up = self.circuit_template.copy()
            bind_up = {self.theta[j]: up[j] for j in range(self.n_qubits)}
            qc_up.bind_parameters(bind_up)
            qc_up.measure_all()
            job_up = execute(qc_up, self.backend, shots=self.shots)
            exp_up_counts = job_up.result().get_counts(qc_up)
            exp_up_val = sum(
                (1 if b == "0" else -1) * c for b, c in exp_up_counts.items()
            ) / (self.shots * self.n_qubits)

            # Shift down
            down = angles.copy()
            down[i] -= shift
            qc_down = self.circuit_template.copy()
            bind_down = {self.theta[j]: down[j] for j in range(self.n_qubits)}
            qc_down.bind_parameters(bind_down)
            qc_down.measure_all()
            job_down = execute(qc_down, self.backend, shots=self.shots)
            exp_down_counts = job_down.result().get_counts(qc_down)
            exp_down_val = sum(
                (1 if b == "0" else -1) * c for b, c in exp_down_counts.items()
            ) / (self.shots * self.n_qubits)

            grads[i] = (exp_up_val - exp_down_val) / 2.0

        return grads
