"""Hybrid classical model combining a CNN feature extractor, a quantum‑inspired random layer, and a configurable classifier head.

The architecture mirrors the quantum counterpart: the CNN extracts image features, a linear random projection emulates the quantum random layer, and a depth‑controlled fully‑connected network implements the classifier.  The design allows easy comparison of classical and quantum performance while sharing the same hyper‑parameter interface.

"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Tuple, List


def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Construct a fully‑connected classifier mirroring the quantum factory.

    Parameters
    ----------
    num_features : int
        Size of the feature vector fed into the classifier.
    depth : int
        Number of hidden layers.

    Returns
    -------
    Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]
        * classifier network
        * list of input indices (encoding) – here simply a range
        * list of parameter counts for each layer
        * list of output indices (observables) – here a single output
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


class HybridNATModel(nn.Module):
    """
    Classical hybrid model that reproduces the structure of the quantum‑NAT architecture.

    The model consists of:
    1. A 2‑layer CNN for feature extraction.
    2. A linear random projection that mimics the quantum random layer.
    3. A depth‑controlled fully‑connected classifier head.
    """

    def __init__(self, depth: int = 2, num_features: int = 4, random_seed: int | None = None) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Random linear layer with orthogonal initialization to emulate random quantum circuit
        self.random_proj = nn.Linear(16 * 7 * 7, num_features, bias=False)
        nn.init.orthogonal_(self.random_proj.weight)

        # Classifier head
        self.classifier, _, self.param_counts, _ = build_classifier_circuit(num_features, depth)

        self.norm = nn.BatchNorm1d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input image of shape (B, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Normalized logits of shape (B, 2).
        """
        bsz = x.shape[0]
        feat = self.features(x)
        flattened = feat.view(bsz, -1)
        proj = self.random_proj(flattened)
        logits = self.classifier(proj)
        return self.norm(logits)

__all__ = ["HybridNATModel"]
