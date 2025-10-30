"""Hybrid classical classifier that combines an embedding, self‑attention, and deep feed‑forward layers."""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Iterable, Tuple, List


class ClassicalSelfAttention(nn.Module):
    """Self‑attention module operating on feature embeddings."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        # Linear projections for Q, K, V
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape (batch, embed_dim)
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        scores = torch.softmax(q @ k.transpose(-1, -2) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ v


def build_classifier_circuit(num_features: int, depth: int,
                             attention_dim: int = 4) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Construct a hybrid classifier that first projects input features to an embedding,
    applies a classical self‑attention block, then a stack of dense layers.
    Returns:
        - model: nn.Sequential
        - encoding: list of feature indices used for encoding
        - weight_sizes: list of parameter counts per layer (including attention)
        - observables: list of output class indices
    """
    encoding = list(range(num_features))
    layers: List[nn.Module] = []

    # Initial linear projection to embedding space
    proj = nn.Linear(num_features, attention_dim)
    layers.append(proj)
    layers.append(nn.ReLU())

    # Self‑attention block
    attention = ClassicalSelfAttention(attention_dim)
    layers.append(attention)
    layers.append(nn.ReLU())

    # Deep feed‑forward stack
    in_dim = attention_dim
    weight_sizes: List[int] = []
    for _ in range(depth):
        linear = nn.Linear(in_dim, attention_dim)
        layers.append(linear)
        layers.append(nn.ReLU())
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = attention_dim

    # Final classification head
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    model = nn.Sequential(*layers)

    # Include attention parameter count
    attn_params = (
        attention.q_linear.weight.numel()
        + attention.k_linear.weight.numel()
        + attention.v_linear.weight.numel()
    )
    weight_sizes.insert(0, attn_params)  # after projection

    observables = [0, 1]  # class indices
    return model, encoding, weight_sizes, observables


__all__ = ["build_classifier_circuit", "ClassicalSelfAttention"]
