import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ClassicalPatchFilter(nn.Module):
    """
    Classical 2×2 patch extractor implemented with a single Conv2d layer.
    Mirrors the patch‑level processing of the quantum quanvolution filter.
    """
    def __init__(self):
        super().__init__()
        # 1 input channel → 4 output channels (each channel corresponds
        # to one pixel of the 2×2 patch).  No bias – the operation is
        # purely linear.
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Flattened patch features of shape (batch, 4*14*14).
        """
        patches = self.conv(x)                     # (batch, 4, 14, 14)
        return patches.view(x.size(0), -1)         # (batch, 4*14*14)


class HybridQuantumClassifier(nn.Module):
    """
    Fully classical binary classifier that mirrors the hybrid quantum
    architecture from the anchor file.  The quantum expectation head is
    replaced by a differentiable sigmoid, preserving the output shape.
    """
    def __init__(self):
        super().__init__()
        # Patch‑wise feature extractor
        self.patch_filter = ClassicalPatchFilter()
        # Simple linear head
        self.fc = nn.Linear(4 * 14 * 14, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing a probability distribution over two classes.

        Parameters
        ----------
        x : torch.Tensor
            Input image of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Probability tensor of shape (batch, 2).
        """
        # Extract features and compute logits
        features = self.patch_filter(x)            # (batch, 4*14*14)
        logits = self.fc(features)                 # (batch, 1)
        probs = torch.sigmoid(logits)              # (batch, 1)
        return torch.cat([probs, 1 - probs], dim=-1)  # (batch, 2)

__all__ = ["HybridQuantumClassifier"]
