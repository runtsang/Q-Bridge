"""
ConvHybrid – classical implementation with a learnable convolution and optional classifier.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Iterable, Tuple, List


def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], List[int], Iterable[int]]:
    """
    Construct a feed‑forward classifier and metadata similar to the quantum variant.

    Parameters
    ----------
    num_features : int
        Number of input features to the classifier.
    depth : int
        Number of hidden layers.

    Returns
    -------
    network : nn.Module
        Sequential network ending in a 2‑class head.
    encoding : Iterable[int]
        Dummy encoding indices (placeholder for potential feature mapping).
    weight_sizes : List[int]
        Number of trainable parameters per layer.
    observables : Iterable[int]
        Dummy observable indices (placeholder for quantum measurement).
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables


class ConvHybrid(nn.Module):
    """
    Classical convolutional filter with an optional classifier head.

    The filter emulates the quantum filter by applying a 2×2 Conv2d followed by a sigmoid
    activation.  The mean activation is returned by :meth:`run`.  The optional classifier
    head can be used for binary classification tasks.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the convolution kernel.
    threshold : float, default 0.0
        Threshold applied before sigmoid to mimic quantum thresholding.
    depth : int, default 2
        Depth of the optional classifier.
    """

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0, depth: int = 2) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        # Optional classifier
        self.classifier, self.encoding, self.weight_sizes, self.observables = build_classifier_circuit(
            num_features=1, depth=depth
        )

    def run(self, data: torch.Tensor) -> float:
        """
        Apply the convolution, sigmoid, and return the mean activation.

        Parameters
        ----------
        data : torch.Tensor
            Input tensor of shape (H, W) or (B, 1, H, W).

        Returns
        -------
        float
            Mean sigmoid activation.
        """
        if data.ndim == 2:
            data = data.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()

    def classify(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classifier head.

        Parameters
        ----------
        data : torch.Tensor
            Input features of shape (B, 1).

        Returns
        -------
        torch.Tensor
            Log‑softmax logits of shape (B, 2).
        """
        return self.classifier(data)

    def get_weight_sizes(self) -> List[int]:
        """Return the list of parameter counts per layer."""
        return self.weight_sizes


__all__ = ["ConvHybrid", "build_classifier_circuit"]
