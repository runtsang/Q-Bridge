"""Quantum kernel construction using TorchQuantum ansatz with trainable amplitudes."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict


class KernalAnsatz(tq.QuantumModule):
    """Variational ansatz with trainable rotation amplitudes."""

    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        # Trainable amplitude for each qubit
        self.amplitude = torch.nn.Parameter(torch.rand(n_wires))

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        """Encode x, decode y with opposite sign."""
        q_device.reset_states(x.shape[0])
        for i in range(self.n_wires):
            params = self.amplitude[i] * x[:, i] if tq.op_name_dict["ry"].num_params else None
            func_name_dict["ry"](q_device, wires=i, params=params)
        # Entangling layer
        for i in range(self.n_wires - 1):
            func_name_dict["cnot"](q_device, wires=[i, i + 1])
        for i in reversed(range(self.n_wires)):
            params = -self.amplitude[i] * y[:, i] if tq.op_name_dict["ry"].num_params else None
            func_name_dict["ry"](q_device, wires=i, params=params)


class Kernel(tq.QuantumModule):
    """Quantum kernel that evaluates the overlap of two encoded states."""

    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(self.n_wires)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the absolute value of the overlap."""
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

    def fit(self, X: torch.Tensor, max_iter: int = 200, lr: float = 0.01) -> "Kernel":
        """Train the amplitude parameters to maximise self‑overlap."""
        optimizer = torch.optim.Adam(self.ansatz.parameters(), lr=lr)
        for _ in range(max_iter):
            optimizer.zero_grad()
            K = self.forward(X, X)
            loss = -torch.mean(torch.diag(K))
            loss.backward()
            optimizer.step()
        return self

    def transform(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute the kernel matrix between two batches."""
        K = torch.zeros((X.size(0), Y.size(0)), device=X.device, dtype=X.dtype)
        for i, x in enumerate(X):
            K[i] = self.forward(x.unsqueeze(0), Y).squeeze(0)
        return K


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor],
                  n_wires: int = 4, subset_size: int | None = None) -> np.ndarray:
    """Compute the Gram matrix via the quantum kernel.

    Parameters
    ----------
    a, b : Sequence[torch.Tensor]
        Collections of 1‑D tensors.
    n_wires : int, optional
        Number of qubits used in the ansatz.
    subset_size : int, optional
        If provided, only a random subset of ``a`` is used.
    """
    kernel = Kernel(n_wires)
    if subset_size is not None:
        idx = np.random.choice(len(a), subset_size, replace=False)
        a = [a[i] for i in idx]
    return np.array([[kernel(x, y).item() for y in b] for x in a])


__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix"]
