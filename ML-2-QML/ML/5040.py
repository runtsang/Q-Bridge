"""Enhanced hybrid CNN + kernel classifier for binary image classification.

This module unifies the classical CNN backbone with an RBF kernel feature map
and a lightweight classifier head.  It is a direct evolution of the
original ClassicalQuantumBinaryClassification.py, incorporating
the FastEstimator utilities for batch evaluation and the kernel construction
from QuantumKernelMethod.py.  The architecture supports efficient
training and inference on CPUs/GPUs.

"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ----------------------------------------------------
# Utility: build_classifier_circuit (classical variant)
# ----------------------------------------------------
def build_classifier_circuit(num_features: int, depth: int) -> tuple[nn.Module, list[int], list[int], list[int]]:
    """Construct a feed‑forward classifier mirroring the quantum helper.

    Parameters
    ----------
    num_features : int
        Dimensionality of the kernel feature vector.
    depth : int
        Number of hidden layers.

    Returns
    -------
    network : nn.Module
        Sequential model ending in a 2‑way classifier.
    encoding : list[int]
        Indices of the input features (identity mapping).
    weight_sizes : list[int]
        Sizes of each linear layer (for bookkeeping).
    observables : list[int]
        Output indices (0, 1).
    """
    layers = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes = []
    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.append(linear)
        layers.append(nn.ReLU(inplace=True))
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())
    network = nn.Sequential(*layers)
    observables = [0, 1]
    return network, encoding, weight_sizes, observables

# ----------------------------------------------------
# RBF kernel feature map
# ----------------------------------------------------
class RBFKernelLayer(nn.Module):
    """Trainable RBF kernel that maps inputs into a high‑dimensional feature space."""
    def __init__(self, input_dim: int, num_centers: int, gamma: float = 1.0):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_centers, input_dim))
        self.gamma = gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, input_dim)
        # Compute pairwise squared distances between x and centers
        diff = x.unsqueeze(1) - self.centers.unsqueeze(0)  # (batch, num_centers, input_dim)
        dist_sq = torch.sum(diff * diff, dim=-1)           # (batch, num_centers)
        return torch.exp(-self.gamma * dist_sq)            # (batch, num_centers)

# ----------------------------------------------------
# Main hybrid kernel classifier
# ----------------------------------------------------
class HybridKernelClassifier(nn.Module):
    """CNN backbone + RBF kernel + linear classifier for binary image tasks."""
    def __init__(
        self,
        num_features: int = 3,
        depth: int = 2,
        num_centers: int = 64,
        gamma: float = 1.0,
    ) -> None:
        super().__init__()
        # Convolutional feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(num_features, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),
            nn.Flatten(),
            nn.Linear(55815, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
        )
        self.kernel_layer = RBFKernelLayer(84, num_centers, gamma)
        self.classifier, _, _, _ = build_classifier_circuit(num_centers, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        kernel_features = self.kernel_layer(features)
        logits = self.classifier(kernel_features)
        return F.softmax(logits, dim=-1)

__all__ = ["HybridKernelClassifier", "build_classifier_circuit", "RBFKernelLayer"]
