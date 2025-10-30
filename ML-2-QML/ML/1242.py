"""
ConvEnhanced: multi‑scale, learnable‑threshold convolutional filter.

Provides a unified interface that can operate in pure classical mode or a hybrid mode that mixes a quantum feature with the classical output. The class is drop‑in compatible with the original `Conv()` API.
"""

from __future__ import annotations

import torch
from torch import nn
import numpy as np


class ConvEnhanced(nn.Module):
    """
    Multi‑scale convolutional filter with learnable thresholds.

    Parameters
    ----------
    mode : str, default="torch"
        Execution mode: "torch" for pure classical, "hybrid" for weighted
        combination of classical and quantum outputs.
    kernel_sizes : int or list[int], default=2
        Size(s) of the convolutional kernels.
    thresholds : float or list[float], default=0.0
        Threshold value(s) applied after the sigmoid activation.
    weight : float, default=0.5
        Weight for the quantum contribution in hybrid mode (0.0–1.0).
    """

    def __init__(
        self,
        mode: str = "torch",
        kernel_sizes: int | list[int] = 2,
        thresholds: float | list[float] = 0.0,
        weight: float = 0.5,
    ):
        super().__init__()
        self.mode = mode
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes]
        if isinstance(thresholds, (int, float)):
            thresholds = [thresholds] * len(kernel_sizes)
        self.kernel_sizes = kernel_sizes
        self.thresholds = nn.ParameterList(
            [nn.Parameter(torch.tensor(t, dtype=torch.float32)) for t in thresholds]
        )
        self.weight_param = nn.Parameter(torch.tensor(weight, dtype=torch.float32))
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, 1, kernel_size=k, bias=True) for k in kernel_sizes]
        )

    def run(self, data: torch.Tensor | np.ndarray) -> float:
        """
        Forward pass.

        Parameters
        ----------
        data : torch.Tensor or np.ndarray
            Input image of shape (H, W) or (1, H, W).

        Returns
        -------
        float
            Aggregated activation value.
        """
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32)
        if data.ndim == 2:
            data = data.unsqueeze(0).unsqueeze(0)
        elif data.ndim == 3:
            data = data.unsqueeze(0)

        activations = []
        for conv, thresh in zip(self.convs, self.thresholds):
            logits = conv(data)
            act = torch.sigmoid(logits - thresh)
            activations.append(act.mean())

        classical = torch.stack(activations).mean()
        if self.mode == "torch":
            return classical.item()
        elif self.mode == "hybrid":
            # Placeholder for quantum output; in practice this would be
            # obtained by calling the QML counterpart.
            quantum = torch.tensor(0.0)
            hybrid = self.weight_param * quantum + (1.0 - self.weight_param) * classical
            return hybrid.item()
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")


def Conv() -> ConvEnhanced:
    """Return a ConvEnhanced instance in classical mode."""
    return ConvEnhanced(mode="torch")
