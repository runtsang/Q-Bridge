"""Hybrid convolutional filter with a quantum kernel.

The :class:`Gen152Conv` module keeps the original Conv filter API while
adding a quantum kernel matrix computation.  It uses Qiskit to evaluate
a parameterized circuit that encodes the data as rotation angles.
"""

from __future__ import annotations

import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit
from qiskit import execute, Aer
import torch
from torch import nn
from typing import Optional, Any


class Gen152Conv(nn.Module):
    """Drop‑in replacement for the original Conv filter with a Qiskit kernel.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the convolutional kernel.
    threshold : float, default 0.0
        Threshold applied after the convolution before the sigmoid.
    shots : int, default 100
        Number of shots for the quantum simulation.
    backend : qiskit.providers.Backend, optional
        Backend used for execution.  Defaults to the Aer qasm simulator.
    """

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0, shots: int = 100,
                 backend: Optional[Any] = None) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        # Quantum kernel circuit
        self.n_qubits = kernel_size * kernel_size
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

        self.register_buffer("reference", torch.empty((0, 1, kernel_size, kernel_size)))

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the convolution and sigmoid activation."""
        if data.ndim == 3:
            data = data.unsqueeze(0)
        conv_out = self.conv(data)
        activations = torch.sigmoid(conv_out - self.threshold)
        return activations

    def set_reference(self, ref: torch.Tensor) -> None:
        """Store a reference set for kernel evaluation."""
        self.reference = ref

    def _run_single(self, data: np.ndarray) -> float:
        """Run the quantum kernel circuit for a single data point."""
        bind = {}
        for i, val in enumerate(data.flatten()):
            bind[self.theta[i]] = np.pi if val > self.threshold else 0.0
        job = execute(self._circuit, self.backend, shots=self.shots,
                      parameter_binds=[bind])
        result = job.result().get_counts(self._circuit)
        # Compute average number of |1⟩ across all qubits
        total = 0
        for bitstring, cnt in result.items():
            total += cnt * bitstring.count("1")
        return total / (self.shots * self.n_qubits)

    def kernel_matrix(self, x: torch.Tensor) -> np.ndarray:
        """Compute the quantum kernel matrix between ``x`` and the stored reference.

        Parameters
        ----------
        x : torch.Tensor
            Convolutional activations of shape ``(N, 1, H, W)``.

        Returns
        -------
        np.ndarray
            Kernel matrix of shape ``(N, M)`` where ``M`` is the number of
            reference samples.
        """
        if self.reference.size(0) == 0:
            raise ValueError("Reference set not set.")
        x_np = x.detach().cpu().numpy()
        ref_np = self.reference.detach().cpu().numpy()
        n_x = x_np.shape[0]
        n_ref = ref_np.shape[0]
        kernel = np.zeros((n_x, n_ref))
        for i in range(n_x):
            for j in range(n_ref):
                kernel[i, j] = self._run_single(x_np[i]) * self._run_single(ref_np[j])
        return kernel


def Conv(kernel_size: int = 2, threshold: float = 0.0, shots: int = 100,
         backend: Optional[Any] = None) -> Gen152Conv:
    """Return a :class:`Gen152Conv` instance.

    This wrapper preserves the original ``Conv`` function name so that
    existing code can import ``Conv`` from :mod:`Conv__gen152` without
    modification.
    """
    return Gen152Conv(kernel_size=kernel_size, threshold=threshold,
                      shots=shots, backend=backend)


__all__ = ["Conv", "Gen152Conv"]
