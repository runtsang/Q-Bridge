"""Hybrid classical classifier combining self‑attention and feed‑forward layers."""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np


class ClassicalSelfAttention(nn.Module):
    """Simple self‑attention block suitable for small feature sets."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = torch.softmax(q @ k.transpose(-2, -1) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ v


def build_classifier_circuit(
    num_features: int,
    depth: int,
    attention_depth: int = 1,
) -> tuple[nn.Module, list[int], list[int], list[int]]:
    """
    Construct a hybrid classical classifier.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input.
    depth : int
        Number of feed‑forward layers.
    attention_depth : int, optional
        Number of self‑attention blocks (default 1).

    Returns
    -------
    network : nn.Module
        Sequential model.
    encoding : list[int]
        Feature indices used for data upload (identity mapping).
    weight_sizes : list[int]
        Number of trainable parameters per linear layer.
    observables : list[int]
        Dummy observables – indices of the output logits.
    """
    layers: list[nn.Module] = []

    # Self‑attention block(s)
    for _ in range(attention_depth):
        layers.append(ClassicalSelfAttention(num_features))

    # Feed‑forward body
    in_dim = num_features
    weight_sizes: list[int] = []
    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.append(linear)
        layers.append(nn.ReLU())
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    # Classification head
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    encoding = list(range(num_features))
    observables = [0, 1]
    return network, encoding, weight_sizes, observables


__all__ = ["build_classifier_circuit", "ClassicalSelfAttention"]
