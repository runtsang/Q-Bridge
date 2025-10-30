"""Hybrid convolutional filter with optional classical classifier.

This module defines a drop‑in replacement for the original `Conv` seed.
The class exposes:
* **feature extraction** – a 2‑D convolution followed by a sigmoid
  activation that can be used as a feature map.
* **classification** – a small feed‑forward network that mirrors the
  quantum classifier interface, enabling end‑to‑end training.
* **scaling** – the same class can be instantiated with a different
  `kernel_size`, `threshold` or `classifier_depth`, allowing it to
  serve as a building block for larger CNNs or as a standalone
  feature extractor.

The design deliberately mirrors the quantum side: the return type,
method names, and signature are identical, so that a downstream
pipeline can switch between classical and quantum back‑ends with
minimal code changes.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Iterable, Tuple, List

# --------------------------------------------------------------------------- #
# Core convolutional filter
# --------------------------------------------------------------------------- #
class ConvFilter(nn.Module):
    """A classical 2‑D convolution filter that emulates the quantum filter.

    Parameters
    ----------
    kernel_size : int
        Size of the convolution kernel.
    threshold : float
        Activation threshold applied before the sigmoid.
    classifier_depth : int, optional
        Number of hidden layers in the optional classifier head.
        If ``None`` the filter returns only the feature map.
    """

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0,
                 classifier_depth: int | None = None) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        # Optional classifier head
        self.classifier: nn.Module | None = None
        if classifier_depth is not None:
            self.classifier = _build_classical_classifier(
                in_features=kernel_size * kernel_size,
                hidden_size=kernel_size * kernel_size,
                depth=classifier_depth
            )

    # ----------------------------------------------------------------------- #
    # Forward pass – returns a scalar activation
    # ----------------------------------------------------------------------- #
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Run the convolution and return the mean sigmoid activation.

        Parameters
        ----------
        data : torch.Tensor
            Shape (H, W); the kernel-sized patch to be processed.

        Returns
        -------
        torch.Tensor
            A single‑value tensor representing the filter response.
        """
        x = data.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean()

    # ----------------------------------------------------------------------- #
    # Feature map extraction
    # ----------------------------------------------------------------------- #
    def feature_map(self, data: torch.Tensor) -> torch.Tensor:
        """Return the raw convolution output (before sigmoid)."""
        x = data.view(1, 1, self.kernel_size, self.kernel_size)
        return self.conv(x)

    # ----------------------------------------------------------------------- #
    # Optional classification
    # ----------------------------------------------------------------------- #
    def classify(self, data: torch.Tensor) -> torch.Tensor:
        """Pass the feature map through the classifier head.

        Raises
        ------
        RuntimeError
            If the filter was instantiated without a classifier head.
        """
        if self.classifier is None:
            raise RuntimeError("Classifier head not configured.")
        fmap = self.feature_map(data).view(-1)  # flatten
        return self.classifier(fmap)

# --------------------------------------------------------------------------- #
# Helper: build a shallow feed‑forward classifier
# --------------------------------------------------------------------------- #
def _build_classical_classifier(in_features: int, hidden_size: int,
                                depth: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_dim = in_features
    for _ in range(depth):
        layers.extend([nn.Linear(in_dim, hidden_size), nn.ReLU()])
        in_dim = hidden_size
    layers.append(nn.Linear(in_dim, 2))  # binary output
    return nn.Sequential(*layers)

# --------------------------------------------------------------------------- #
# Public factory – mirrors the quantum interface
# --------------------------------------------------------------------------- #
def Conv(kernel_size: int = 2, threshold: float = 0.0,
         classifier_depth: int | None = None) -> ConvFilter:
    """Return a classical convolutional filter with optional classifier.

    Parameters
    ----------
    kernel_size : int, optional
        Size of the convolution kernel.
    threshold : float, optional
        Activation threshold.
    classifier_depth : int, optional
        Depth of the optional feed‑forward classifier.

    Returns
    -------
    ConvFilter
        An instance ready to be called on a 2‑D patch.
    """
    return ConvFilter(kernel_size=kernel_size,
                      threshold=threshold,
                      classifier_depth=classifier_depth)

__all__ = ["Conv", "ConvFilter"]
