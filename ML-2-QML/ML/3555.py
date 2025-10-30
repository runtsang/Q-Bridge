"""Hybrid classical classifier with optional self‑attention layer.

The module mirrors the public interface of the original
`QuantumClassifierModel.build_classifier_circuit` but augments it with a
learnable self‑attention block.  This provides richer feature
transformation before the feed‑forward classifier, improving
expressivity while staying fully classical.

The returned tuple matches the original signature:
    (model, encoding, weight_sizes, observables)

The `encoding` list is kept for API compatibility and simply contains the
indices of the input features.  `observables` are a placeholder that can
be used for downstream evaluation (e.g. probability distributions).
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import math

def _build_self_attention(embed_dim: int) -> nn.Module:
    """Construct a lightweight self‑attention layer with learnable
    rotation and entanglement matrices.  The implementation follows
    the classical SelfAttention helper from the reference pair but
    converts it into a PyTorch `nn.Module` so it can be trained end‑to‑end.
    """
    class SelfAttentionLayer(nn.Module):
        def __init__(self, embed_dim: int):
            super().__init__()
            self.embed_dim = embed_dim
            # rotation and entanglement parameters are square matrices
            self.rotation = nn.Parameter(torch.randn(embed_dim, embed_dim))
            self.entangle = nn.Parameter(torch.randn(embed_dim, embed_dim))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, embed_dim)
            query = x @ self.rotation
            key = x @ self.entangle
            scores = torch.softmax(query @ key.T / math.sqrt(self.embed_dim), dim=-1)
            return scores @ x

    return SelfAttentionLayer(embed_dim)

class HybridClassifier(nn.Module):
    """A classical feed‑forward classifier enriched with a self‑attention
    mechanism.  It can be instantiated with any number of hidden layers
    (depth).  The `build_classifier_circuit` helper returns all metadata
    required for reproducibility and for aligning with the quantum
    counterpart.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vector.
    depth : int
        Number of hidden layers in the classifier.
    embed_dim : int, optional
        Dimensionality of the self‑attention sub‑space.  Defaults to
        ``num_features`` for a full‑rank attention.
    """
    def __init__(self, num_features: int, depth: int, embed_dim: int | None = None):
        super().__init__()
        self.num_features = num_features
        self.depth = depth
        self.embed_dim = embed_dim or num_features

        layers: List[nn.Module] = []
        # self‑attention block
        self.attention = _build_self_attention(self.embed_dim)
        layers.append(self.attention)

        # feed‑forward layers
        in_dim = self.embed_dim
        for _ in range(depth):
            linear = nn.Linear(in_dim, self.num_features)
            layers.append(linear)
            layers.append(nn.ReLU())
            in_dim = self.num_features
        # output head
        layers.append(nn.Linear(in_dim, 2))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """Instantiate a hybrid classical classifier and return training metadata.

    The signature matches the original ML helper to keep API compatibility.
    The returned `weight_sizes` list contains the number of learnable
    parameters for each linear layer; the self‑attention parameters are
    omitted for brevity, but can be added if needed.

    Parameters
    ----------
    num_features : int
        Input dimensionality.
    depth : int
        Number of hidden layers.

    Returns
    -------
    model : nn.Module
        The constructed hybrid classifier.
    encoding : list[int]
        Placeholder list of input feature indices.
    weight_sizes : list[int]
        Number of parameters per linear layer.
    observables : list[int]
        Placeholder observable identifiers.
    """
    model = HybridClassifier(num_features, depth)
    encoding = list(range(num_features))
    weight_sizes = []
    for module in model.network:
        if isinstance(module, nn.Linear):
            weight_sizes.append(module.weight.numel() + module.bias.numel())
    observables = list(range(2))
    return model, encoding, weight_sizes, observables

__all__ = ["build_classifier_circuit", "HybridClassifier"]
