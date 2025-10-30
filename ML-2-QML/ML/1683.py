"""Hybrid classical neural network inspired by the original quanvolution.

Features:
- Learnable patch‑wise linear mapping (quantum‑style feature extractor).
- Classical CNN backbone for further spatial reasoning.
- Fully differentiable and trainable with standard PyTorch optimizers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionHybrid(nn.Module):
    """
    Classical implementation of the quanvolution idea.
    The first stage maps each 2×2 patch to a 4‑dimensional feature vector
    using a trainable linear layer.  The resulting feature map of shape
    (batch, 4, 14, 14) is processed by a small CNN backbone followed
    by a classifier.
    """
    def __init__(self, in_channels: int = 1, n_classes: int = 10) -> None:
        super().__init__()
        # Patch‑wise linear mapping: 4 inputs → 4 outputs
        self.patch_mapper = nn.Linear(4, 4, bias=True)

        # CNN backbone that consumes the 4‑channel feature map
        self.backbone = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 7×7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 3×3
            nn.Flatten()
        )

        # Compute flattened feature size: 64 * 3 * 3 = 576
        self.classifier = nn.Linear(576, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor of shape (batch, 1, 28, 28)

        Returns:
            log‑softmax over classes
        """
        batch_size = x.shape[0]
        # Extract 2×2 patches
        patches = x.unfold(2, 2, 2).unfold(3, 2, 2)  # shape: B, 1, 14, 14, 2, 2
        patches = patches.contiguous().view(batch_size, 14, 14, 4)  # B, 14, 14, 4
        patches = patches.view(batch_size, -1, 4)  # B, 196, 4

        # Map each patch to 4‑dim feature vector
        mapped = self.patch_mapper(patches)  # B, 196, 4
        mapped = mapped.view(batch_size, 4, 14, 14)  # B, 4, 14, 14

        # Backbone
        features = self.backbone(mapped)  # B, 576

        logits = self.classifier(features)  # B, n_classes
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionHybrid"]
