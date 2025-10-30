"""Hybrid classical classifier combining CNN, self‑attention and graph‑aware feed‑forward layers."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Iterable, Tuple, List

# --------------------------------------------------------------------------- #
#  Feature extractor – identical to the Quantum‑NAT CNN
# --------------------------------------------------------------------------- #
class _CNNFeatureExtractor(nn.Module):
    def __init__(self, out_dim: int = 64) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(nn.Linear(16 * 7 * 7, out_dim), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.fc(x.view(x.size(0), -1))

# --------------------------------------------------------------------------- #
#  Classical self‑attention block
# --------------------------------------------------------------------------- #
class _SelfAttention(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.Wq = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.Wk = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.Wv = nn.Parameter(torch.randn(embed_dim, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = x @ self.Wq
        k = x @ self.Wk
        v = x @ self.Wv
        scores = F.softmax(q @ k.transpose(-2, -1) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ v

# --------------------------------------------------------------------------- #
#  Graph‑aware feed‑forward network
# --------------------------------------------------------------------------- #
class _GraphFeedforward(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int], adjacency: torch.Tensor) -> None:
        super().__init__()
        self.adj = adjacency
        layers = []
        dims = [in_dim] + hidden_dims
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_d, out_d))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simple message passing: weighted sum of neighbours
        x = x @ self.adj
        return self.net(x)

# --------------------------------------------------------------------------- #
#  Utility: build adjacency from fidelity threshold
# --------------------------------------------------------------------------- #
def _build_fidelity_adjacency(
    states: Iterable[torch.Tensor], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5
) -> torch.Tensor:
    n = len(states)
    adj = torch.zeros((n, n), device=next(iter(states)).device)
    for i, a in enumerate(states):
        for j, b in enumerate(states[i + 1:], i + 1):
            fid = torch.dot(a, b) / (torch.norm(a) * torch.norm(b) + 1e-12)
            if fid >= threshold:
                adj[i, j] = adj[j, i] = 1.0
            elif secondary is not None and fid >= secondary:
                adj[i, j] = adj[j, i] = secondary_weight
    return adj

# --------------------------------------------------------------------------- #
#  Hybrid classifier – public API
# --------------------------------------------------------------------------- #
class HybridClassifier(nn.Module):
    """
    Classical hybrid classifier that first extracts convolutional features,
    then applies a self‑attention layer, and finally a graph‑aware feed‑forward
    network.  The adjacency matrix is constructed from a set of prototype
    states via a fidelity threshold.
    """

    def __init__(
        self,
        num_features: int,
        depth: int,
        embed_dim: int,
        graph_arch: List[int],
        prototype_states: List[torch.Tensor],
        fidelity_threshold: float,
        *,
        secondary_threshold: float | None = None,
        secondary_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.feature_extractor = _CNNFeatureExtractor(out_dim=num_features)
        self.attention = _SelfAttention(embed_dim=embed_dim)
        adj = _build_fidelity_adjacency(
            prototype_states, fidelity_threshold,
            secondary=secondary_threshold, secondary_weight=secondary_weight,
        )
        self.classifier = _GraphFeedforward(num_features, graph_arch, adj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = self.attention(x)
        return self.classifier(x)

# --------------------------------------------------------------------------- #
#  Function that mimics the quantum interface
# --------------------------------------------------------------------------- #
def build_classifier_circuit(
    num_features: int,
    depth: int,
    embed_dim: int,
    graph_arch: List[int],
    prototype_states: List[torch.Tensor],
    fidelity_threshold: float,
    *,
    secondary_threshold: float | None = None,
    secondary_weight: float = 0.5,
) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Return the hybrid classifier together with metadata that mirrors the
    quantum interface: encoding indices, weight sizes and observables.
    """
    model = HybridClassifier(
        num_features=num_features,
        depth=depth,
        embed_dim=embed_dim,
        graph_arch=graph_arch,
        prototype_states=prototype_states,
        fidelity_threshold=fidelity_threshold,
        secondary_threshold=secondary_threshold,
        secondary_weight=secondary_weight,
    )
    # Dummy metadata – the quantum side will use these placeholders
    encoding = list(range(num_features))
    weight_sizes = [p.numel() for p in model.parameters()]
    observables = [0]  # placeholder
    return model, encoding, weight_sizes, observables

__all__ = ["HybridClassifier", "build_classifier_circuit"]
