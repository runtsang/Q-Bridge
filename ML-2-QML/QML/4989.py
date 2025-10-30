from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

import qiskit
from qiskit import execute
from qiskit.circuit import Parameter
from qiskit.circuit.random import random_circuit


class ConvGen256(nn.Module):
    """
    Quantum implementation of the multi‑scale convolutional filter.
    The module uses a quantum circuit to compute expectation values
    over 2×2 image patches and can be extended to larger kernels
    by sliding windows.  It remains drop‑in compatible with the
    classical ConvGen256.
    """
    def __init__(
        self,
        kernel_size: int = 2,
        stride: int = 1,
        padding: int = 0,
        threshold: float = 0.0,
        shots: int = 1024,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.threshold = threshold
        self.shots = shots
        # Quantum backend
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        # Build a quantum circuit that encodes a patch of size kernel_size×kernel_size
        n_qubits = kernel_size ** 2
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = [Parameter(f"theta{i}") for i in range(n_qubits)]
        # Encode pixel values via Ry rotations
        for i in range(n_qubits):
            self.circuit.ry(self.theta[i], i)
        # Add a small random entangling layer
        self.circuit.barrier()
        if n_qubits > 1:
            self.circuit += random_circuit(n_qubits, 2)
        # Measure all qubits
        self.circuit.measure_all()

    def _run_circuit(self, patch_flat: np.ndarray) -> float:
        """
        Execute the quantum circuit for a single flattened patch.
        """
        param_bind = {
            self.theta[i]: np.pi if val > self.threshold else 0.0
            for i, val in enumerate(patch_flat)
        }
        job = execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[param_bind],
        )
        result = job.result()
        # Compute expectation value of Z on all qubits
        counts = result.get_counts(self.circuit)
        total_ones = 0
        total_counts = 0
        for state, cnt in counts.items():
            ones = sum(int(b) for b in state)
            total_ones += ones * cnt
            total_counts += cnt
        return total_ones / (total_counts * len(patch_flat))

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            data: Tensor of shape (H, W) or (1, 1, H, W).

        Returns:
            Tensor containing the mean quantum expectation over all patches,
            shape (1,).
        """
        if data.ndim == 2:
            data = data.unsqueeze(0).unsqueeze(0)
        elif data.ndim == 3:
            data = data.unsqueeze(0)
        # Sliding window extraction
        patches = []
        H, W = data.shape[2], data.shape[3]
        for i in range(0, H - self.kernel_size + 1, self.stride):
            for j in range(0, W - self.kernel_size + 1, self.stride):
                patch = data[:, :, i : i + self.kernel_size, j : j + self.kernel_size]
                patches.append(patch.view(-1).cpu().numpy())
        if not patches:
            return torch.tensor(0.0, dtype=torch.float32)
        expectations = [self._run_circuit(p) for p in patches]
        return torch.tensor(np.mean(expectations), dtype=torch.float32)
