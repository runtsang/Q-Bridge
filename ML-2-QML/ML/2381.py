"""Hybrid convolutional classifier for classical experiments.

This module implements a drop‑in replacement for the original Conv.py
by combining a learnable convolutional pre‑processor with a
multi‑layer feed‑forward classifier.  The class exposes a simple
``run`` interface that accepts a 2‑D image patch and returns a
classification probability.  The design mirrors the quantum
counterpart, enabling a side‑by‑side comparison of classical and
quantum performance.

The architecture is intentionally modular:
  * ``self.conv`` – a 2‑D convolution that learns a kernel of size
    ``kernel_size``.
  * ``self.classifier`` – a fully‑connected network of ``depth`` layers
    operating on the scalar output of the convolution.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Tuple


class HybridConvClassifier(nn.Module):
    """Classical convolution + feed‑forward classifier."""

    def __init__(
        self,
        kernel_size: int = 2,
        depth: int = 2,
        num_features: int = 10,
        threshold: float = 0.0,
        device: str | torch.device = "cpu",
    ) -> None:
        """
        Parameters
        ----------
        kernel_size : int
            Size of the square convolutional kernel.
        depth : int
            Number of hidden layers in the classifier.
        num_features : int
            Width of the hidden layers.
        threshold : float
            Bias applied before the sigmoid non‑linearity.
        device : str or torch.device
            Target device for tensors.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.device = torch.device(device)

        # Convolutional pre‑processor
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            bias=True,
        ).to(self.device)

        # Feed‑forward classifier
        layers: list[nn.Module] = []
        in_dim = 1  # scalar output from conv
        for _ in range(depth):
            linear = nn.Linear(in_dim, num_features).to(self.device)
            layers.extend([linear, nn.ReLU()])
            in_dim = num_features
        # Final layer to 2 classes
        head = nn.Linear(in_dim, 2).to(self.device)
        layers.append(head)

        self.classifier = nn.Sequential(*layers).to(self.device)

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the convolution and classifier.

        Parameters
        ----------
        patch : torch.Tensor
            Tensor of shape ``(B, 1, H, W)`` where ``H == W == kernel_size``.
            The batch dimension ``B`` may be omitted; the method handles
            both batched and single‑sample inputs.

        Returns
        -------
        torch.Tensor
            Log‑softmax probabilities of shape ``(B, 2)``.
        """
        # Ensure patch is on the correct device
        patch = patch.to(self.device)

        # Convolution
        logits = self.conv(patch)
        activations = torch.sigmoid(logits - self.threshold)
        # Reduce spatial dimensions to a scalar per sample
        features = activations.view(activations.size(0), -1).mean(dim=1, keepdim=True)

        # Classifier
        logits = self.classifier(features)
        probs = torch.softmax(logits, dim=1)
        return probs

    def run(self, data: torch.Tensor) -> Tuple[float, float]:
        """
        Convenience wrapper that accepts a NumPy array or torch tensor
        and returns a tuple of class probabilities.

        Parameters
        ----------
        data : torch.Tensor or numpy.ndarray
            Image patch of shape ``(kernel_size, kernel_size)`` or
            ``(B, kernel_size, kernel_size)``.

        Returns
        -------
        Tuple[float, float]
            Probabilities for classes 0 and 1.
        """
        if not isinstance(data, torch.Tensor):
            import numpy as np

            data = torch.as_tensor(data, dtype=torch.float32)
        if data.ndim == 2:
            data = data.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif data.ndim == 3 and data.shape[0] == 1:
            data = data.unsqueeze(1)  # (1, 1, H, W)
        elif data.ndim == 4 and data.shape[1] == 1:
            pass  # already (B, 1, H, W)
        else:
            raise ValueError("Unsupported input shape for HybridConvClassifier.run")

        probs = self.forward(data)
        return tuple(probs.squeeze().tolist())


__all__ = ["HybridConvClassifier"]
