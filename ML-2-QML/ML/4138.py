from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionHybrid(nn.Module):
    """
    Classical implementation of the hybrid quanvolution architecture.
    Uses a 2×2 convolution with sigmoid activation to emulate the
    quantum patchwise filter and a linear head for classification.
    """

    def __init__(self, num_classes: int = 10, threshold: float = 0.0) -> None:
        super().__init__()
        # 2×2 convolution that reduces 28×28 to 14×14 feature maps
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=True)
        self.threshold = threshold
        # Linear head mapping 4×14×14 features to class logits
        self.fc = nn.Linear(4 * 14 * 14, num_classes)
        self.bn = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input image batch of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Log‑softmax logits of shape (batch, num_classes).
        """
        features = self.conv(x)  # (batch, 4, 14, 14)
        features = torch.sigmoid(features - self.threshold)  # quantum‑style sigmoid
        features = features.view(x.size(0), -1)  # flatten
        logits = self.fc(features)
        logits = self.bn(logits)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionHybrid"]
