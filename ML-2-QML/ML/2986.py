"""Hybrid classical convolutional classifier.

The module defines a `HybridConvClassifier` that combines a classical
convolutional filter with a feed‑forward network.  The design mirrors
the quantum counterpart but is fully NumPy/Torch based, enabling rapid
experimentation on CPUs/GPUs.  The class exposes a `run` method that
accepts a 2‑D array and returns the softmax probability for the
positive class.

The architecture is inspired by the two reference pairs:
- The convolutional block borrows the kernel‑size, thresholding and
  sigmoid activation from the original Conv.py.
- The classifier construction follows the depth‑controlled network
  from QuantumClassifierModel.py, but replaces quantum observables
  with a linear head.  The returned `weight_sizes` provide a
  convenient metric for model complexity.

A small helper `build_classifier_circuit` is exposed so that the
class can be instantiated with arbitrary depth and feature size.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import Iterable, Tuple, List


def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Construct a feed‑forward classifier and metadata similar to the quantum variant.
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: List[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU(inplace=True)])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables


class HybridConvClassifier(nn.Module):
    """
    Classical analogue of the quantum convolution + classifier stack.

    Parameters
    ----------
    kernel_size : int
        Size of the square kernel used in the convolutional filter.
    threshold : float
        Value used to threshold input pixels before sigmoid activation.
    num_features : int
        Dimensionality of the linear classifier input.
    depth : int
        Number of hidden layers in the classifier.
    """
    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        num_features: int = 4,
        depth: int = 2,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        self.classifier, _, _, _ = build_classifier_circuit(num_features, depth)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the filter and classifier.

        Parameters
        ----------
        data : torch.Tensor
            2‑D tensor with shape (kernel_size, kernel_size).

        Returns
        -------
        torch.Tensor
            Logits from the classifier.
        """
        tensor = data.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        features = activations.view(-1)  # flatten
        return self.classifier(features)

    def predict(self, data: torch.Tensor) -> torch.Tensor:
        """
        Return the probability of the positive class.

        Parameters
        ----------
        data : torch.Tensor
            2‑D tensor with shape (kernel_size, kernel_size).

        Returns
        -------
        torch.Tensor
            Probability of class 1.
        """
        logits = self.forward(data)
        probs = torch.softmax(logits, dim=-1)
        return probs[:, 1]


__all__ = ["HybridConvClassifier", "build_classifier_circuit"]
