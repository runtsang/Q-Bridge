"""Classical self‑attention classifier mirroring the quantum interface."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Tuple, List


def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """Construct a feed‑forward classifier with a self‑attention block.

    Returns:
        network: nn.Module implementing attention + depth‑wise FFN + head.
        encoding: list of indices used for data encoding.
        weight_sizes: list of parameter counts per layer.
        observables: list of output class indices.
    """
    class ClassicalSelfAttentionClassifier(nn.Module):
        def __init__(self, embed_dim: int, depth: int):
            super().__init__()
            self.embed_dim = embed_dim
            self.depth = depth
            self.attn = nn.MultiheadAttention(embed_dim, num_heads=1, batch_first=True)
            ffn_layers = []
            for _ in range(depth):
                ffn_layers.extend([nn.Linear(embed_dim, embed_dim), nn.ReLU()])
            self.ffn = nn.Sequential(*ffn_layers)
            self.classifier = nn.Linear(embed_dim, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape: (batch, seq_len, embed_dim)
            attn_output, _ = self.attn(x, x, x)
            ffn_output = self.ffn(attn_output)
            logits = self.classifier(ffn_output.mean(dim=1))
            return logits

    network = ClassicalSelfAttentionClassifier(embed_dim=num_features, depth=depth)
    encoding = list(range(num_features))
    weight_sizes = [p.numel() for p in network.parameters()]
    observables = [0, 1]
    return network, encoding, weight_sizes, observables


__all__ = ["build_classifier_circuit"]
