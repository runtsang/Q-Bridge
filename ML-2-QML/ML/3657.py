"""Hybrid classical filter and classifier with quantum‑inspired feature mapping.

The class can operate in two modes:
  - ``classical``: uses a standard 2×2 Conv2d followed by a sigmoid threshold.
  - ``quantum``: applies a learnable linear transformation to 2×2 patches,
    mimicking a quantum feature map for efficient training without quantum
    hardware.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionHybrid(nn.Module):
    """
    Classical or quantum‑inspired quanvolution filter followed by a linear head.
    Parameters
    ----------
    mode : str, optional
        ``'classical'`` or ``'quantum'``.  Default ``'classical'``.
    threshold : float, optional
        Threshold applied before sigmoid in classical mode.  Default ``0.0``.
    num_filters : int, optional
        Number of output channels / features per patch.  Default ``4``.
    """

    def __init__(
        self,
        mode: str = "classical",
        threshold: float = 0.0,
        num_filters: int = 4,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.threshold = threshold
        self.num_filters = num_filters

        if mode == "classical":
            self.conv = nn.Conv2d(
                in_channels=1,
                out_channels=num_filters,
                kernel_size=2,
                stride=2,
                bias=True,
            )
            self.activator = nn.Sigmoid()
        else:  # quantum‑inspired
            # Linear layer to emulate a quantum feature map
            self.quantum_layer = nn.Linear(4, num_filters, bias=True)
            nn.init.orthogonal_(self.quantum_layer.weight)
            nn.init.constant_(self.quantum_layer.bias, 0.0)

        # Classifier head
        self.classifier = nn.Linear(num_filters * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Log‑softmax logits of shape (B, 10).
        """
        if self.mode == "classical":
            features = self.conv(x)
            features = self.activator(features - self.threshold)
            # Shape: (B, num_filters, 14, 14)
        else:  # quantum‑inspired
            # Extract non‑overlapping 2×2 patches
            patches = (
                x.unfold(2, 2, 2).unfold(3, 2, 2)  # (B, C, H', W', 2, 2)
               .contiguous()
               .view(x.size(0), -1, 4)  # (B, N, 4)
            )
            # Map each patch to feature space
            features = self.quantum_layer(patches)  # (B, N, num_filters)
            # Permute to (B, num_filters, N) then flatten
            features = features.permute(0, 2, 1).contiguous()
            # Shape: (B, num_filters, 14*14)

        flat = features.view(x.size(0), -1)  # (B, num_filters*14*14)
        logits = self.classifier(flat)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionHybrid"]
