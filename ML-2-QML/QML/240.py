"""Quantum quanvolutional filter and classifier with adjustable depth and residual connections."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class Quanvolution__gen266(tq.QuantumModule):
    """
    Quantum quanvolutional filter and classifier with adjustable depth and residual connections.
    The filter extracts 2×2 patches, encodes them into a 4‑qubit register, applies a
    depth‑controlled variational circuit, and measures all qubits. A residual
    connection is implemented via a classical 1×1 convolution on the input,
    followed by a 2×2 max‑pool to match feature dimensions.
    """
    def __init__(self, depth: int = 1, device: str = "cpu") -> None:
        """
        Args:
            depth: Number of variational layers applied to each patch.
            device: The device for the quantum simulation ('cpu', 'gpu', etc.).
        """
        super().__init__()
        self.depth = depth
        self.device = device
        self.n_wires = 4

        # Encoder: map each pixel to a rotation on a qubit
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

        # Residual classical 1×1 conv and pooling
        self.residual_conv = nn.Conv2d(1, 4, kernel_size=1, bias=False)
        self.residual_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Variational circuit: repeated depth times
        self.var_circuit = tq.RandomLayer(n_ops=depth * 4, wires=list(range(self.n_wires)))

        self.measure = tq.MeasureAll(tq.PauliZ)

        # Classifier head
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)

        # Residual path (classical)
        residual = self.residual_conv(x)
        residual = self.residual_pool(residual)
        residual_flat = residual.view(bsz, -1)

        # Prepare patches
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.var_circuit(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))

        out = torch.cat(patches, dim=1)

        # Add residual
        out = out + residual_flat[:, :out.shape[1]]

        # Classify
        logits = self.linear(out)
        return F.log_softmax(logits, dim=-1)

__all__ = ["Quanvolution__gen266"]
