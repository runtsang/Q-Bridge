"""Hybrid classical layer that aggregates fully‑connected, convolutional, quanvolution, and Quantum‑NAT inspired blocks."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Iterable, Optional

__all__ = ["HybridQuantumLayer"]


class _Conv2DFilter(nn.Module):
    """2×2 classical convolution used to emulate a quantum filter."""
    def __init__(self, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=kernel_size, stride=stride, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _QuanvolutionFilter(nn.Module):
    """Classical replacement for a quantum quanvolution: 2×2 patches → 4‑dim features."""
    def __init__(self, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.Conv2d(1, 4, kernel_size=kernel_size, stride=stride, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _QNatHead(nn.Module):
    """Mimics a Quantum‑NAT fully‑connected head with a small MLP."""
    def __init__(self, in_features: int, hidden: int = 64, out_features: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
            nn.BatchNorm1d(out_features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HybridQuantumLayer(nn.Module):
    """
    A drop‑in replacement for the original FCL that unifies:
      * a 2×2 quantum‑style fully‑connected block,
      * a classical convolutional filter,
      * a quanvolution filter,
      * a Quantum‑NAT inspired linear head.
    The `run` method accepts either a tensor of parameters for the fully‑connected block
    or a 4‑channel image; it returns a 4‑dimensional output.
    """
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        # Fully‑connected quantum emulation
        self.fc = nn.Linear(n_features, 1, bias=True)

        # Classical convolutional front‑end
        self.conv = _Conv2DFilter()
        # Classical quanvolution front‑end
        self.quanv = _QuanvolutionFilter()
        # Quantum‑NAT inspired head
        self.qnat = _QNatHead(in_features=4 * 14 * 14)  # assuming 28×28 input

    def _fc_expectation(self, thetas: Iterable[float]) -> torch.Tensor:
        """Return the mean tanh activation of the linear layer."""
        theta_tensor = torch.tensor(list(thetas), dtype=torch.float32, device=self.fc.weight.device)
        if theta_tensor.ndim == 0:
            theta_tensor = theta_tensor.unsqueeze(0)
        out = torch.tanh(self.fc(theta_tensor))
        return out.mean()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward: image → conv → quanv → flatten → QNat head."""
        # Assume input shape: (batch, 1, 28, 28)
        conv_out = self.conv(x)          # (batch, 4, 14, 14)
        quanv_out = self.quanv(x)        # (batch, 4, 14, 14)
        # Merge by addition (simple fusion)
        fused = conv_out + quanv_out
        flat = fused.view(fused.size(0), -1)  # (batch, 4*14*14)
        return self.qnat(flat)

    def run(self, data, mode: str = "fc") -> torch.Tensor:
        """
        Unified run interface.
        * mode='fc'  → interpret `data` as iterable of parameters for the FC block.
        * mode='img' → interpret `data` as a single image and return the full forward output.
        """
        if mode == "fc":
            return self._fc_expectation(data)
        elif mode == "img":
            return self.forward(data)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
