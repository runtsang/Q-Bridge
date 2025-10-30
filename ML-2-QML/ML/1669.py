"""Enhanced classical classifier builder with optional 1‑D convolutional feature extraction and weight introspection.

The `build_classifier_circuit` function returns a `torch.nn.Sequential` network, the indices of encoded features,
a list of parameter counts per layer, and a list of observables (here simply `[0, 1]` for binary classification).
The API stays identical to the original seed but adds an optional `conv` flag and `dropout` to facilitate
robust feature learning.

Example
-------
>>> net, enc, wts, obs = build_classifier_circuit(num_features=64, depth=4, conv=True, dropout=0.3)
>>> print(net)
Sequential(...)
"""
from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
from torch import nn


def build_classifier_circuit(
    num_features: int,
    depth: int,
    conv: bool = False,
    dropout: float = 0.0,
) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Construct a feed‑forward classifier with an optional 1‑D CNN front‑end.

    Parameters
    ----------
    num_features:
        Number of input features.
    depth:
        Number of fully‑connected layers in the classifier head.
    conv:
        Whether to prepend a shallow 1‑D convolutional block.
    dropout:
        Dropout probability applied after each linear layer (if > 0).

    Returns
    -------
    network:
        A `torch.nn.Sequential` model ready for training.
    encoding:
        List of input feature indices (identity mapping).
    weight_sizes:
        List with the number of trainable parameters per layer.
    observables:
        List of class indices, kept for API compatibility.
    """
    layers: List[nn.Module] = []

    # Optional 1‑D convolutional feature extractor
    if conv:
        conv_block = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        layers.append(conv_block)

    in_dim = num_features if not conv else 32 * num_features  # approximate output dim
    weight_sizes: List[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.append(linear)
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.ReLU())
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    encoding = list(range(num_features))
    observables = [0, 1]
    return network, encoding, weight_sizes, observables
