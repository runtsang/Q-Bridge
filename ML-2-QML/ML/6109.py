"""Hybrid classical‑quantum convolutional filter.

This module provides a drop‑in replacement for the original Conv filter.
It contains a learnable classical kernel and an optional quantum
variational circuit that can be trained jointly. The outputs of the two
branches are combined via a learnable fusion weight.

Usage:
    from ConvEnhanced import ConvEnhanced
    conv = ConvEnhanced(kernel_size=3, use_quantum=True)
    out = conv(torch.randn(4,1,3,3))
"""

from __future__ import annotations

import torch
from torch import nn
import pennylane as qml
import numpy as np


class ConvEnhanced(nn.Module):
    """Hybrid classical‑quantum convolutional filter."""

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        *,
        use_quantum: bool = True,
        quantum_shots: int = 100,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.use_quantum = use_quantum

        # Classical sub‑module: a single learnable 2‑D kernel
        self.classical = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            bias=True,
        )

        # Quantum sub‑module: a Pennylane device and circuit
        self.quantum_size = kernel_size ** 2
        self.quantum_device = qml.device(
            "default.qubit",
            wires=self.quantum_size,
            shots=quantum_shots,
        )
        # Parameters for the variational part
        self.qparams = nn.Parameter(torch.randn(self.quantum_size))

        # Build the circuit
        self._build_circuit()

        # Fusion weight: learnable scalar that blends the two outputs
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

    def _build_circuit(self):
        """Define a parametric circuit that acts on the data qubits."""

        @qml.qnode(self.quantum_device, interface="torch")
        def circuit(x: torch.Tensor):
            # Encode data as rotations around X
            for i in range(self.quantum_size):
                qml.RX(x[..., i], wires=i)
            # Simple entangling layer
            for i in range(self.quantum_size - 1):
                qml.CNOT(wires=(i, i + 1))
            # Parameterised rotations
            for i in range(self.quantum_size):
                qml.RY(self.qparams[i], wires=i)
            # Return expectation values of Z on all wires
            return [qml.expval(qml.PauliZ(i)) for i in range(self.quantum_size)]

        self.circuit = circuit

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Return a fused output from classical and quantum parts.

        Args:
            data: Tensor of shape (batch, 1, kernel_size, kernel_size).
        """
        # Classical path
        class_out = self.classical(data).mean()

        # Quantum path
        if self.use_quantum:
            # Flatten data to (batch, n_qubits)
            batch_size = data.shape[0]
            x = data.view(batch_size, self.quantum_size)
            # Run the circuit; output shape (batch, n_qubits)
            q_out = self.circuit(x)  # list of tensors
            q_out = torch.stack(q_out, dim=-1)  # shape (batch, n_qubits)
            quantum_out = q_out.mean()
        else:
            quantum_out = torch.tensor(0.0, device=data.device)

        # Fuse with learnable weight
        return self.fusion_weight * class_out + (1.0 - self.fusion_weight) * quantum_out


__all__ = ["ConvEnhanced"]
