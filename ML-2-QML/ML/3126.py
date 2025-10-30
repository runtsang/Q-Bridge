"""Hybrid convolutional filter with optional quantum‑inspired initialization.

This module defines a drop‑in replacement for the original ``Conv`` class
in the anchor repository.  The design draws from the classical example
and the quantum seed to provide a single, extensible API that can be
used in both pure‑classical and hybrid settings.

Features
--------
- 2×2 convolution with bias and ReLU activation.
- Optional thresholding that emulates the quantum filter’s output range.
- Weight initialization can optionally use expectation values from a
  small random quantum circuit (via qiskit) to seed the classical
  parameters with quantum‑inspired statistics.
- ``run`` method mirrors the quantum counterpart, returning the mean
  activation as a scalar.

Usage
-----
>>> from Conv__gen051 import ConvGen051
>>> conv = ConvGen051(kernel_size=2, threshold=0.5, init_from_qc=True)
>>> data = torch.randn(1, 1, 2, 2)
>>> conv.run(data)
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
try:
    # Optional quantum initialization
    import qiskit
    from qiskit.circuit.random import random_circuit
except ImportError:
    qiskit = None

class ConvGen051(nn.Module):
    """
    Classical 2×2 convolution with thresholding and optional quantum‑inspired
    weight initialization.
    """
    def __init__(self,
                 kernel_size: int = 2,
                 threshold: float = 0.0,
                 init_from_qc: bool = False,
                 qc_shots: int = 200) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        if init_from_qc and qiskit is not None:
            self._qc_init(qc_shots)

    def _qc_init(self, shots: int) -> None:
        """Use a small random circuit to generate expectation‑based weights."""
        n_qubits = self.kernel_size ** 2
        circuit = qiskit.QuantumCircuit(n_qubits)
        circuit.h(range(n_qubits))
        circuit.barrier()
        circuit += random_circuit(n_qubits, 2)
        circuit.measure_all()

        backend = qiskit.Aer.get_backend("qasm_simulator")
        job = qiskit.execute(circuit, backend, shots=shots)
        result = job.result()
        counts = result.get_counts(circuit)
        probs = np.array([counts.get(bin(i)[2:].zfill(n_qubits), 0) / shots
                          for i in range(2**n_qubits)])
        # Map probabilities to weight matrix
        weights = probs.reshape((1, 1, self.kernel_size, self.kernel_size))
        self.conv.weight.data = torch.tensor(weights, dtype=torch.float32)

    def run(self, data: torch.Tensor) -> float:
        """
        Forward pass with thresholding.

        Parameters
        ----------
        data : torch.Tensor
            Input tensor of shape (N, 1, H, W).  For compatibility with the
            original API, only the first sample is used.

        Returns
        -------
        float
            Mean activation after applying the threshold.
        """
        out = self.conv(data)
        activated = F.relu(out - self.threshold)
        return activated.mean().item()

__all__ = ["ConvGen051"]
