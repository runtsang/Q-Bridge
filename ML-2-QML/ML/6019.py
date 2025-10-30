"""Enhanced classical classifier with residual connections and flexible preprocessing."""

from __future__ import annotations

from typing import Iterable, Tuple, List, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):
    """A simple residual block with two linear layers and a ReLU."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(out_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return F.relu(x + residual)


def build_classifier_circuit(
    num_features: int,
    depth: int,
    hidden_dim: int = 64,
    use_residual: bool = True,
) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Construct a feed‑forward classifier with optional residual blocks.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vector.
    depth : int
        Number of hidden layers (or residual blocks).
    hidden_dim : int, optional
        Width of each hidden layer. Default is 64.
    use_residual : bool, optional
        If True, each layer is wrapped in a ResidualBlock. Default is True.

    Returns
    -------
    nn.Module
        The constructed network.
    Iterable[int]
        List of indices used to encode the input (identity mapping).
    Iterable[int]
        List of weight‑size counts for each trainable layer.
    List[int]
        List of output class indices (two‑class classification).
    """
    layers: List[nn.Module] = []
    in_dim = num_features

    # Identity encoding: we expose the indices of the raw features
    encoding = list(range(num_features))

    # Build hidden layers
    for _ in range(depth):
        linear = nn.Linear(in_dim, hidden_dim)
        if use_residual:
            layers.append(ResidualBlock(in_dim, hidden_dim))
        else:
            layers.append(linear)
            layers.append(nn.ReLU())
        in_dim = hidden_dim

    # Classification head
    head = nn.Linear(in_dim, 2)
    layers.append(head)

    # Assemble the sequential model
    network = nn.Sequential(*layers)

    # Compute weight sizes for reporting
    weight_sizes = []
    for layer in network:
        if isinstance(layer, nn.Linear) or isinstance(layer, ResidualBlock):
            weight_sizes.append(layer.linear1.weight.numel() + layer.linear1.bias.numel())
            if isinstance(layer, ResidualBlock):
                weight_sizes.append(layer.linear2.weight.numel() + layer.linear2.bias.numel())
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    # Binary classification observables
    observables = [0, 1]

    return network, encoding, weight_sizes, observables


def standardize_batch(
    X: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize a batch of feature vectors in place, optionally computing mean/std.

    Parameters
    ----------
    X : np.ndarray
        Input feature matrix (batch, features).
    mean : np.ndarray | None
        Pre‑computed mean vector. If None, it is computed from X.
    std : np.ndarray | None
        Pre‑computed standard deviation vector. If None, it is computed from X.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Standardized data, mean, and std.
    """
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0) + 1e-8
    X_std = (X - mean) / std
    return X_std, mean, std


__all__ = ["build_classifier_circuit", "standardize_batch"]
