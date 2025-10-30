"""Quantum‑dual Quanvolution using Pennylane."""

from __future__ import annotations

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane.numpy as np
from typing import List

class QuanvolutionDual(nn.Module):
    """Hybrid model with a quantum convolution branch and a classical branch.
    The quantum branch processes each 2×2 patch with a parameterized circuit
    and produces an amplitude‑based feature vector. The outputs are concatenated
    with the classical convolution features before the final classifier.
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        q_layers: int = 2,
        q_wires: int = 4,
        conv_out_channels: int = 4,
        conv_kernel_size: int = 2,
        conv_stride: int = 2,
    ) -> None:
        super().__init__()
        # Quantum circuit configuration
        self.q_layers = q_layers
        self.q_wires = q_wires
        self.dev = qml.device("default.qubit", wires=self.q_wires)
        self.q_circuit = qml.QNode(self._q_circuit, self.dev)

        # Classical convolution branch
        self.conv_branch = nn.Conv2d(
            in_channels,
            conv_out_channels,
            kernel_size=conv_kernel_size,
            stride=conv_stride,
        )

        # Fusion head
        conv_feat_dim = conv_out_channels * 14 * 14
        q_feat_dim = q_wires * 14 * 14
        self.fc = nn.Linear(conv_feat_dim + q_feat_dim, num_classes)

    def _q_circuit(self, data: np.ndarray):
        """Parameterized circuit acting on a 2×2 patch encoded as rotations."""
        for i, angle in enumerate(data):
            qml.RY(angle, wires=i)
        for _ in range(self.q_layers):
            for w in range(self.q_wires - 1):
                qml.CNOT(wires=[w, w + 1])
            qml.ry(np.pi / 4, wires=self.q_wires - 1)
        return [qml.expval(qml.PauliZ(i)) for i in range(self.q_wires)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical branch
        conv_feat = self.conv_branch(x)
        conv_flat = conv_feat.view(x.size(0), -1)

        # Quantum branch: process each 2×2 patch
        q_patches: List[torch.Tensor] = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = x[:, r:r+2, c:c+2].squeeze(1)  # (batch, 2, 2)
                patch_flat = patch.view(x.size(0), -1)  # (batch, 4)
                q_out_batch: List[torch.Tensor] = []
                for i in range(x.size(0)):
                    q_out = self.q_circuit(patch_flat[i].cpu().numpy())
                    q_out_batch.append(torch.tensor(q_out, device=x.device, dtype=torch.float32))
                q_patch = torch.stack(q_out_batch, dim=0)  # (batch, q_wires)
                q_patches.append(q_patch)
        q_feat = torch.cat(q_patches, dim=1)  # (batch, num_patches * q_wires)

        # Concatenate and classify
        fused = torch.cat([conv_flat, q_feat], dim=1)
        logits = self.fc(fused)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionDual"]
