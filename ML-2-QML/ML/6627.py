"""Classical convolutional filter with optional quantum post‑processing.

The module defines a `HybridConv` class that implements a 2‑D convolution followed by a sigmoid
threshold.  An optional quantum classifier can be attached via `set_quantum_classifier`.  The
class remains fully classical (PyTorch / NumPy) and can be used as a drop‑in replacement for
the original `Conv.py` while still supporting hybrid experiments.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Optional, Callable, Any

class HybridConv(nn.Module):
    """
    Classical convolution followed by an optional quantum classifier.

    Parameters
    ----------
    kernel_size : int, default=2
        Size of the convolution kernel.
    threshold : float, default=0.0
        Threshold applied after the sigmoid activation.
    """

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        self._quantum_classifier: Optional[Callable[[torch.Tensor], float]] = None

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classical convolution.

        Parameters
        ----------
        data : torch.Tensor
            Input tensor of shape (H, W).

        Returns
        -------
        torch.Tensor
            Convolved and thresholded tensor of shape (1, 1, H-k+1, W-k+1).
        """
        tensor = data.unsqueeze(0).unsqueeze(0)  # shape (1, 1, H, W)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations

    def run(self, data: Any) -> float:
        """
        Execute the full hybrid pipeline.

        Parameters
        ----------
        data : array‑like or torch.Tensor
            2‑D input array.

        Returns
        -------
        float
            If a quantum classifier is attached, its output; otherwise the mean activation.
        """
        # Ensure we have a torch tensor
        if not isinstance(data, torch.Tensor):
            data = torch.as_tensor(data, dtype=torch.float32)

        activations = self.forward(data)
        mean_activation = activations.mean().item()

        if self._quantum_classifier is None:
            return mean_activation

        # Flatten the activations to feed into the quantum classifier
        flat = activations.view(-1)
        return float(self._quantum_classifier(flat))

    def set_quantum_classifier(self, classifier: Callable[[torch.Tensor], float]) -> None:
        """
        Attach a quantum classifier to the hybrid pipeline.

        Parameters
        ----------
        classifier : Callable[[torch.Tensor], float]
            Function that accepts a 1‑D tensor and returns a scalar.
        """
        self._quantum_classifier = classifier

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(kernel_size={self.kernel_size}, threshold={self.threshold}, quantum_attached={self._quantum_classifier is not None})"

__all__ = ["HybridConv"]
