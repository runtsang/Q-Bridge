"""Quantum kernel construction with a parameter‑optimized variational ansatz."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from torch import nn


class VariationalAnsatz(tq.QuantumModule):
    """Parameterized ansatz with trainable rotation angles.

    The ansatz applies a single layer of Ry rotations with parameters
    that are linear functions of the input features. The weights are
    learnable and are updated during training.
    """

    def __init__(self, n_wires: int, hidden_dim: int = 8) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Linear layers to map input features to rotation angles
        self.angle_mlp = nn.Sequential(
            nn.Linear(n_wires, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_wires)
        )
        # Register trainable parameters for each wire (optional)
        self.trainable_offsets = nn.Parameter(torch.zeros(n_wires))

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Encode x and y into the quantum device and compute overlap.

        Args:
            q_device: QuantumDevice instance.
            x: Tensor of shape (batch, n_wires)
            y: Tensor of shape (batch, n_wires)
        """
        # Reset device
        q_device.reset_states(x.shape[0])
        # Encode x
        angles_x = self.angle_mlp(x) + self.trainable_offsets
        for wire in range(self.n_wires):
            func_name_dict["ry"](q_device, wires=[wire], params=angles_x[:, wire])
        # Encode y with negative angles
        angles_y = self.angle_mlp(y) + self.trainable_offsets
        for wire in range(self.n_wires):
            func_name_dict["ry"](q_device, wires=[wire], params=-angles_y[:, wire])


class QuantumKernel(tq.QuantumModule):
    """Quantum kernel implemented with a variational ansatz."""

    def __init__(self, n_wires: int = 4, hidden_dim: int = 8) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = VariationalAnsatz(self.n_wires, hidden_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the kernel matrix between two batches of inputs.

        Args:
            x: Tensor of shape (n_samples_x, n_wires)
            y: Tensor of shape (n_samples_y, n_wires)

        Returns:
            Tensor of shape (n_samples_x, n_samples_y)
        """
        n_x = x.shape[0]
        n_y = y.shape[0]
        kernel_vals = torch.empty((n_x, n_y), device=x.device)
        for i in range(n_x):
            for j in range(n_y):
                self.ansatz(self.q_device, x[i:i+1], y[j:j+1])
                # Overlap is amplitude of |0...0>
                kernel_vals[i, j] = torch.abs(self.q_device.states.view(-1)[0])
        return kernel_vals


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], hidden_dim: int = 8) -> np.ndarray:
    """
    Compute the Gram matrix between two sequences of tensors using the quantum kernel.

    Args:
        a: Sequence of tensors of shape (n_wires,)
        b: Sequence of tensors of shape (n_wires,)
        hidden_dim: Hidden dimension of the variational ansatz.

    Returns:
        NumPy array of shape (len(a), len(b))
    """
    kernel = QuantumKernel(n_wires=a[0].shape[0], hidden_dim=hidden_dim)
    a_batch = torch.stack(a)
    b_batch = torch.stack(b)
    with torch.no_grad():
        return kernel(a_batch, b_batch).cpu().numpy()


__all__ = ["VariationalAnsatz", "QuantumKernel", "kernel_matrix"]
