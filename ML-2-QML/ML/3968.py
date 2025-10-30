"""Hybrid Quantum Classifier - Classical implementation.

This module provides a drop‑in replacement that emulates the quantum
pipeline using classical convolutional filtering followed by a
feed‑forward neural network.  It preserves the API of the original
QuantumClassifierModel while offering a scalable, fully classical
alternative suitable for large‑scale experiments.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from.Conv import Conv as ClassicalConv

__all__ = ["HybridQuantumClassifier", "build_classifier_circuit"]

def build_classifier_circuit(num_features: int, depth: int) -> nn.Module:
    """Construct a simple feed‑forward classifier.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vector.
    depth : int
        Number of hidden layers.
    """
    layers = []
    in_dim = num_features
    for _ in range(depth):
        layers.append(nn.Linear(in_dim, num_features))
        layers.append(nn.ReLU())
        in_dim = num_features
    layers.append(nn.Linear(in_dim, 2))
    return nn.Sequential(*layers)

class HybridQuantumClassifier(nn.Module):
    """Classical stand‑in for the quantum classifier.

    The class first applies a 2‑D convolutional filter (emulating the
    quantum quanvolution layer) and then feeds the extracted feature
    map to a variationally‑shaped neural network defined by
    ``build_classifier_circuit``.
    """

    def __init__(
        self,
        num_features: int,
        depth: int,
        conv_kernel: int = 2,
        conv_threshold: float = 0.0,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.conv = ClassicalConv(kernel_size=conv_kernel, threshold=conv_threshold)
        self.network = build_classifier_circuit(num_features, depth)
        self.device = device
        self.to(device)

    def run(self, data: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Apply convolutional preprocessing and predict class logits.

        Parameters
        ----------
        data : torch.Tensor or np.ndarray
            2‑D array of shape (H, W) or a pre‑flattened 1‑D vector
            of length ``num_features``.
        """
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32)
        data = data.to(self.device)

        # If the input is 2‑D, run the convolutional filter over all
        # non‑overlapping patches and flatten the result.
        if data.ndim == 2:
            patch_size = self.conv.kernel_size
            H, W = data.shape
            features = []
            for i in range(0, H - patch_size + 1, patch_size):
                for j in range(0, W - patch_size + 1, patch_size):
                    patch = data[i : i + patch_size, j : j + patch_size]
                    features.append(self.conv.run(patch.cpu().numpy()))
            features = torch.tensor(features, dtype=torch.float32, device=self.device)
        else:
            features = data

        logits = self.network(features)
        return logits
