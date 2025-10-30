"""ConvEnhanced: a hybrid‑classical convolutional filter with adaptive learning."""

from __future__ import annotations

import math
from typing import List

import torch
from torch import nn
import torch.nn.functional as F


class ConvEnhanced(nn.Module):
    """
    A drop‑in replacement for the original Conv filter that supports:
    * Multi‑kernel sizes (1×1, 2×2, 3×3, …)
    * Learnable kernel weights via a tiny MLP
    * Adaptive thresholding based on the running mean of activations
    * Optional fusion with a quantum‑based filter
    """

    def __init__(
        self,
        kernel_sizes: List[int] | None = None,
        mlp_hidden: int = 32,
        threshold_init: float = 0.0,
        use_quantum: bool = False,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.kernel_sizes = kernel_sizes or [2]
        self.threshold = nn.Parameter(torch.tensor(threshold_init, dtype=torch.float32))
        self.mlp = nn.Sequential(
            nn.Linear(len(self.kernel_sizes), mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, len(self.kernel_sizes)),
        )
        self.use_quantum = use_quantum
        self.device = device
        # Pre‑compute a small set of learnable kernels for each size
        self.kernels = nn.ParameterDict()
        for k in self.kernel_sizes:
            self.kernels[f"k{k}"] = nn.Parameter(
                torch.randn(1, 1, k, k) / math.sqrt(k * k)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass over a single image patch.
        Args:
            x: Tensor of shape (batch, 1, H, W) – the patch to apply
              to the conv‑like filter.
        Returns:
            float: mean activation across all kernels.
        """
        # Compute weighted sum of kernels
        weights = torch.softmax(self.mlp(torch.tensor(self.kernel_sizes, device=x.device)), dim=-1)
        outputs = []
        for k, w in zip(self.kernel_sizes, weights):
            conv = F.conv2d(x, self.kernels[f"k{k}"], bias=None, padding=0)
            conv = conv * w
            conv = torch.sigmoid(conv - self.threshold)
            outputs.append(conv)
        # fuse all kernel sizes
        out = torch.stack(outputs, dim=0).mean(dim=0)
        if self.use_quantum:
            # The quantum filter is executed in a separate module
            try:
                from.quantum import ConvQuantum
                quantum_out = ConvQuantum.apply(x, self.threshold.item())
                out = 0.5 * out + 0.5 * quantum_out
            except Exception:
                pass  # fall back to classical output if quantum module is unavailable
        return out.mean().item()

    @torch.no_grad()
    def _update_threshold(self, activations: torch.Tensor) -> None:
        """
        A simple exponential‑moving‑average (EMA) for adaptive threshold.
        """
        mean_act = activations.mean()
        self.threshold.data = 0.8 * self.threshold.data + 0.1 * mean_act


__all__ = ["ConvEnhanced"]
