"""Hybrid kernel classifier – classical implementation.

The module mirrors the quantum interface while remaining fully
classical.  It exposes a :class:`HybridKernelClassifier` that
implements a kernel‑ridge regressor based on an RBF kernel and a
feed‑forward neural network for feature extraction.  The public
API matches the quantum side so that the two modules can be swapped
without changing downstream code.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RBFKernel(nn.Module):
    """Pure‑Python RBF kernel compatible with the quantum interface."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class HybridKernelClassifier(nn.Module):
    """
    Classical hybrid kernel classifier.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input data.
    gamma : float, default 1.0
        RBF kernel width.
    depth : int, default 2
        Depth of the internal feed‑forward network.
    """

    def __init__(self, num_features: int, gamma: float = 1.0, depth: int = 2) -> None:
        super().__init__()
        self.kernel = RBFKernel(gamma)
        self.feature_extractor, self.encoding, self.weight_sizes, self.observables = build_classifier_circuit(
            num_features, depth
        )
        self.alpha: torch.Tensor | None = None
        self.X_train: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feature extractor followed by kernel prediction.
        """
        if self.alpha is None or self.X_train is None:
            raise RuntimeError("Model has not been fitted yet.")
        features = self.feature_extractor(x)
        K = self.kernel(features, self.X_train)
        return torch.matmul(K, self.alpha)

    def fit(self, X: torch.Tensor, y: torch.Tensor, lambda_reg: float = 1e-3) -> None:
        """
        Fit a kernel ridge regressor on the extracted features.

        Parameters
        ----------
        X : torch.Tensor
            Training samples of shape (n_samples, n_features).
        y : torch.Tensor
            Target labels of shape (n_samples,).
        lambda_reg : float
            Ridge regularisation strength.
        """
        # Extract features
        feats = self.feature_extractor(X)
        self.X_train = feats.detach()
        # Compute Gram matrix
        K = self.kernel(feats, feats)
        # Solve (K + λI)α = y
        n = K.shape[0]
        A = K + lambda_reg * torch.eye(n, device=K.device)
        self.alpha = torch.linalg.solve(A, y.to(K.dtype))

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predict labels for new samples.
        """
        return self.forward(X)

    def kernel_matrix(self, a: Iterable[torch.Tensor], b: Iterable[torch.Tensor]) -> np.ndarray:
        """
        Compute the Gram matrix between two collections of tensors.
        """
        return np.array([[self.kernel(x, y).item() for y in b] for x in a])


def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Construct a feed‑forward classifier mirroring the quantum helper interface.

    Returns
    -------
    network : nn.Module
        Sequential network with ReLU activations.
    encoding : Iterable[int]
        Feature indices used for the initial linear layer.
    weight_sizes : Iterable[int]
        Number of trainable parameters in each linear block.
    observables : List[int]
        Dummy observable list kept for API compatibility.
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


__all__ = ["HybridKernelClassifier", "build_classifier_circuit"]
