"""Hybrid classical convolution module with residual attention and quantum fusion.

Designed for use as a drop‑in replacement for the original Conv class.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from qml import QuantumFilter  # assumes the quantum module is available in the same package

__all__ = ["ConvEnhanced"]


class ConvEnhanced(nn.Module):
    """
    ConvEnhanced implements a hybrid convolution with a classical path and a quantum path.
    The classical path applies a learnable 2‑D convolution followed by a residual attention map.
    The quantum path applies a variational circuit whose parameters are updated by a
    parameter‑shift rule during training.  The two outputs are fused by a learnable
    weight that is trained jointly.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        num_qubits: int | None = None,
        backend: str = "qasm_simulator",
        shots: int = 1024,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold

        # Classical branch
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        self.attn = nn.Conv2d(1, 1, kernel_size=1, bias=False)

        # Quantum branch
        if num_qubits is None:
            num_qubits = kernel_size ** 2
        self.quantum = QuantumFilter(
            num_qubits=num_qubits, backend=backend, shots=shots, threshold=threshold
        )

        # Fusion weight
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, 1, H, W) where the patch size is kernel_size.
        Returns:
            Tensor of shape (batch,) with the fused output.
        """
        # Classical path
        cls = self.conv(x)
        cls = torch.sigmoid(cls - self.threshold)
        cls = cls.mean(dim=[2, 3])  # global average
        attn = torch.sigmoid(self.attn(x))
        attn = attn.mean(dim=[2, 3])
        cls = cls * attn

        # Quantum path
        patches = (
            x.unfold(2, self.kernel_size, self.kernel_size)
           .unfold(3, self.kernel_size, self.kernel_size)
        )
        patches = patches.squeeze(1).squeeze(1)  # (batch, k, k)
        with torch.no_grad():
            q_outputs = []
            for i in range(patches.shape[0]):
                arr = patches[i].cpu().numpy()
                q_outputs.append(self.quantum.run(arr))
            q_out = torch.tensor(q_outputs, dtype=torch.float32, device=x.device)

        # Fuse
        out = self.alpha * cls.squeeze() + (1 - self.alpha) * q_out
        return out
