"""Quantum kernel based on a parameter‑shift ansatz using Pennylane.

The original implementation used TorchQuantum.  This version
leverages Pennylane’s automatic differentiation and GPU
back‑end, making it easier to embed in PyTorch models.  The
ansatz consists of Ry rotations followed by a CNOT ladder,
and the kernel is the squared overlap (fidelity) between two
states prepared from inputs `x` and `y`.  The module supports
batched evaluation, automatic device selection, and a
convenient `kernel_matrix` helper.

Key extensions:
* Use of Pennylane’s `qml.qnode` for efficient simulation.
* GPU support via the `default.qubit` device on CUDA.
* Batch‑wise evaluation to avoid memory blow‑up.
* Exposed `kernel_matrix` that accepts torch tensors or lists.
"""
from __future__ import annotations

import numpy as np
import pennylane as qml
import torch
from torch import nn

# Default device: GPU if available, otherwise CPU
DEV = qml.device("default.qubit", wires=4, shots=1024, gpu=torch.cuda.is_available())

def _ry_ansatz(params: torch.Tensor, wires: list[int]) -> None:
    """Apply Ry rotations."""
    for w in wires:
        qml.RY(params[w], wires=w)

def _entangle(wires: list[int]) -> None:
    """CNOT ladder entanglement."""
    for i in range(len(wires)-1):
        qml.CNOT(wires=[wires[i], wires[i+1]])

class HybridKernel(nn.Module):
    """Quantum kernel based on a Ry ansatz and CNOT entanglement."""
    def __init__(self, n_wires: int = 4, device: str | qml.Device = DEV) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.device = device

        @qml.qnode(self.device, interface="torch", diff_method="parameter-shift")
        def _kernel_qnode(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """Prepare two states and return their fidelity."""
            # Encode x
            _ry_ansatz(x, list(range(self.n_wires)))
            _entangle(list(range(self.n_wires)))
            # Apply inverse of y
            _ry_ansatz(-y, list(range(self.n_wires)))
            _entangle(list(range(self.n_wires)))
            # Probability of the all‑zero state equals the squared overlap
            return qml.probs(wires=range(self.n_wires))[0]

        self._qnode = _kernel_qnode

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the squared overlap (fidelity) between |ψ(x)⟩ and |ψ(y)⟩."""
        x = x.to(self.device)
        y = y.to(self.device)
        return self._qnode(x, y)

    def kernel_matrix(self, a: torch.Tensor | list[torch.Tensor],
                      b: torch.Tensor | list[torch.Tensor],
                      batch_size: int = 32) -> np.ndarray:
        """Compute the Gram matrix between two collections of vectors."""
        if isinstance(a, list):
            a = torch.stack(a)
        if isinstance(b, list):
            b = torch.stack(b)
        a = a.to(self.device)
        b = b.to(self.device)

        n, d = a.shape
        m, _ = b.shape
        out = torch.empty((n, m), device=self.device)

        for i in range(0, n, batch_size):
            a_batch = a[i:i+batch_size]
            for j in range(0, m, batch_size):
                b_batch = b[j:j+batch_size]
                for ii, xi in enumerate(a_batch):
                    for jj, yj in enumerate(b_batch):
                        out[i+ii, j+jj] = self.forward(xi, yj)
        return out.cpu().numpy()

def kernel_matrix(a: list[torch.Tensor], b: list[torch.Tensor],
                  n_wires: int = 4, device: str | qml.Device = DEV,
                  batch_size: int = 32) -> np.ndarray:
    """Convenience wrapper that returns a NumPy array."""
    kernel = HybridKernel(n_wires, device)
    return kernel.kernel_matrix(a, b, batch_size)

__all__ = ["HybridKernel", "kernel_matrix"]
