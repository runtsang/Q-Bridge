from __future__ import annotations

import numpy as np
import torch
from torch import nn

class ConvFilter(nn.Module):
    """
    Classical convolutional filter augmented with a small feed‑forward classifier.
    The interface mirrors the original Conv() factory: an instance exposes a run()
    method that accepts a 2‑D NumPy array and returns a classification probability.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0, depth: int = 2) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        # Simple classifier head: flatten conv output → hidden → 2‑class logits
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(kernel_size * kernel_size, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, data: np.ndarray) -> float:
        """
        Forward pass: convolution → sigmoid activation → classification.
        """
        if data.ndim == 2:
            data = data[None, None, :, :]  # add batch & channel dims

        tensor = torch.as_tensor(data, dtype=torch.float32)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        out = self.classifier(activations)
        probs = torch.softmax(out, dim=-1)
        return probs.mean().item()

    def run(self, data: np.ndarray) -> float:
        """
        Compatibility wrapper for the original 'run' API.
        """
        return self.forward(data)

def Conv() -> ConvFilter:
    """
    Factory function compatible with the original Conv.py interface.
    """
    return ConvFilter()

__all__ = ["Conv"]
