"""Classical classifier with attention and deeper network.

The function `build_classifier_circuit` keeps the original signature and
returns a PyTorch model that contains an embedding layer, a single‑head
self‑attention block, a configurable number of linear‑ReLU layers and a
2‑class head.  The helper also returns metadata that mirrors the seed:
`encoding`, `weight_sizes` and a dummy `observables` list.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn


def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """Construct an attention‑based classifier.

    Returns:
        network      – nn.Module implementing the classifier.
        encoding     – list of feature indices (unchanged from the seed).
        weight_sizes – number of trainable parameters for each block.
        observables  – dummy list of observable IDs for compatibility.
    """
    class AttentionClassifier(nn.Module):
        def __init__(self, num_features: int, depth: int):
            super().__init__()
            self.embedding = nn.Linear(num_features, num_features)
            self.attention = nn.MultiheadAttention(embed_dim=num_features,
                                                    num_heads=1,
                                                    batch_first=True)
            self.layers = nn.ModuleList()
            in_dim = num_features
            for _ in range(depth):
                self.layers.append(nn.Linear(in_dim, num_features))
                self.layers.append(nn.ReLU())
                in_dim = num_features
            self.head = nn.Linear(in_dim, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape: (batch, num_features)
            x = self.embedding(x)                       # (batch, num_features)
            x = x.unsqueeze(1)                          # (batch, 1, num_features)
            x, _ = self.attention(x, x, x)              # (batch, 1, num_features)
            x = x.squeeze(1)                            # (batch, num_features)
            for layer in self.layers:
                x = layer(x)
            return self.head(x)

    network = AttentionClassifier(num_features, depth)

    # Count parameters per block
    weight_sizes = [p.numel() for p in network.parameters() if p.requires_grad]

    encoding = list(range(num_features))
    observables = list(range(2))
    return network, encoding, weight_sizes, observables


__all__ = ["build_classifier_circuit"]
