"""Hybrid classifier combining classical convolution with a simulated quantum layer.

The module defines a PyTorch nn.Module that applies a 2×2 convolution to the input,
flattens the feature map, simulates a depth‑controlled quantum circuit using
parameterized rotations, and finally classifies with a linear head.
The quantum layer is a lightweight classical simulation that mirrors the
parameter structure of the Qiskit implementation in the QML seed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

# Classical convolutional filter inspired by Conv.py and Quanvolution.py
class ConvFilter(nn.Module):
    """
    2×2 convolution followed by a sigmoid activation.
    Mirrors the behaviour of the classical quanvolution filter.
    """
    def __init__(self, kernel_size: int = 2, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=kernel_size, stride=stride, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, H, W)
        return self.conv(x)

# Classical simulation of a quantum circuit
class ClassicalQuantumLayer(nn.Module):
    """
    Simulates a depth‑controlled quantum circuit with Rx encoding,
    Ry rotations and CZ entanglement.  The simulation is purely
    classical and operates on the expectation value of Pauli‑Z.
    """
    def __init__(self, num_qubits: int, depth: int) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.depth = depth
        # Encoding parameters (one per qubit)
        self.encoding = nn.Parameter(torch.zeros(num_qubits))
        # Variational parameters (one Ry per qubit per depth)
        self.theta = nn.Parameter(torch.randn(num_qubits * depth))
        # Observables: Pauli‑Z on each qubit
        self.observables = ["Z"] * num_qubits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, num_qubits) – flattened feature vector
        Returns the expectation values of Z for each qubit.
        """
        batch = x.shape[0]
        # Starting expectation of Z in |0> state is 1
        z_vals = torch.ones(batch, self.num_qubits, device=x.device)

        # Encoding: apply Rx(x_i) rotation, update expectation values
        z_vals = torch.cos(self.encoding)  # shape (num_qubits,)
        z_vals = z_vals.unsqueeze(0).expand(batch, -1)

        # Variational layers
        idx = 0
        for _ in range(self.depth):
            # Ry rotations
            ry_vals = torch.cos(self.theta[idx:idx+self.num_qubits])
            idx += self.num_qubits
            z_vals = z_vals * ry_vals  # element‑wise multiplication as a toy model

            # CZ entanglement: decorrelate neighbouring qubits
            z_vals = z_vals * torch.roll(z_vals, shifts=1, dims=1)

        return z_vals

class HybridClassifier(nn.Module):
    """
    Hybrid classical‑quantum classifier.

    Architecture:
        ConvFilter → Flatten → ClassicalQuantumLayer → Linear
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        num_qubits: int = 256,
        depth: int = 4,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.conv = ConvFilter()
        self.flatten = nn.Flatten()
        # After conv: 4 output channels, kernel 2, stride 1 -> (H-1, W-1)
        h, w = input_shape[1] - 1, input_shape[2] - 1
        conv_features = 4 * h * w
        if conv_features!= num_qubits:
            self.pad = nn.ConstantPad1d((0, num_qubits - conv_features), 0)
        else:
            self.pad = None

        self.quantum = ClassicalQuantumLayer(num_qubits, depth)
        self.head = nn.Linear(num_qubits, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, H, W)
        x = self.conv(x)
        x = self.flatten(x)
        if self.pad is not None:
            x = self.pad(x)
        q_out = self.quantum(x)
        logits = self.head(q_out)
        return logits

    def get_metadata(self) -> Tuple[List[int], List[int], List[str]]:
        """
        Return lists of encoding indices, weight sizes of each layer,
        and observable identifiers.
        """
        encoding = list(range(self.quantum.num_qubits))
        weight_sizes = [p.numel() for p in self.quantum.parameters()]
        observables = self.quantum.observables
        return encoding, weight_sizes, observables

__all__ = ["HybridClassifier"]
