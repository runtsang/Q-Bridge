"""Quantum kernel construction using Pennylane and a variational ansatz."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pennylane as qml
import torch
from torch import nn

class QuantumKernelMethod:
    """
    Quantum kernel evaluated via a trainable variational circuit.
    The circuit encodes two input vectors into separate quantum states
    and returns the absolute inner product as the kernel value.
    """

    def __init__(self, n_wires: int = 4, device_name: str = "default.qubit") -> None:
        self.n_wires = n_wires
        self.dev = qml.device(device_name, wires=n_wires)

        # Trainable parameters for the ansatz
        self.ansatz_params = nn.Parameter(torch.randn(n_wires, dtype=torch.float32))

        # Compile the state preparation circuit
        @qml.qnode(self.dev, interface="torch", diff_method="parameter-shift")
        def state_qnode(x: torch.Tensor) -> torch.Tensor:
            # Encode the input data
            for i in range(self.n_wires):
                qml.RY(x[i], wires=i)
            # Apply the variational layer
            for i in range(self.n_wires):
                qml.RY(self.ansatz_params[i], wires=i)
            return qml.state()

        self.state_qnode = state_qnode

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the quantum kernel between two batches of samples.
        x, y: shape (batch, features)
        Returns: shape (batch, batch)
        """
        # Prepare state vectors for each sample
        state_a = torch.stack([self.state_qnode(xi) for xi in x])  # (B, 2**n_wires)
        state_b = torch.stack([self.state_qnode(yi) for yi in y])  # (B, 2**n_wires)

        # Compute the absolute inner product between all pairs
        kernels = torch.abs(torch.matmul(state_a, state_b.conj().t()))  # (B, B)
        return kernels

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """
        Compute the Gram matrix between two sequences of tensors.
        Each element in a and b is a 1‑D tensor of features.
        """
        a_stack = torch.stack(a)  # (N, F)
        b_stack = torch.stack(b)  # (M, F)
        return self.forward(a_stack, b_stack).detach().cpu().numpy()

__all__ = ["QuantumKernelMethod"]
