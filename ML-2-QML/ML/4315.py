"""HybridSelfAttentionClassifier – classical implementation."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalSelfAttention:
    """Pure‑Python self‑attention block mirroring the quantum interface."""
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        # Linear projections
        query = torch.as_tensor(
            inputs @ rotation_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        key = torch.as_tensor(
            inputs @ entangle_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        value = torch.as_tensor(inputs, dtype=torch.float32)
        # Attention scores
        scores = torch.softmax(
            query @ key.T / np.sqrt(self.embed_dim), dim=-1
        )
        return (scores @ value).numpy()


def build_classifier_circuit(num_features: int, depth: int) -> nn.Sequential:
    """Feed‑forward classifier with ReLU layers, mirroring the quantum ansatz."""
    layers = []
    in_dim = num_features
    for _ in range(depth):
        layers.append(nn.Linear(in_dim, num_features))
        layers.append(nn.ReLU())
        in_dim = num_features
    layers.append(nn.Linear(in_dim, 2))
    return nn.Sequential(*layers)


class HybridSelfAttentionClassifier(nn.Module):
    """Classical self‑attention followed by a hybrid classifier."""
    def __init__(self, embed_dim: int = 4, num_features: int = 10, depth: int = 2):
        super().__init__()
        self.self_attention = ClassicalSelfAttention(embed_dim)
        self.classifier = build_classifier_circuit(num_features, depth)

    def forward(
        self,
        inputs: torch.Tensor,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> torch.Tensor:
        # Classical self‑attention
        attn = self.self_attention.run(rotation_params, entangle_params, inputs.numpy())
        attn_tensor = torch.from_numpy(attn).float()
        # Classifier head
        logits = self.classifier(attn_tensor)
        probs = torch.softmax(logits, dim=-1)
        return probs


__all__ = ["HybridSelfAttentionClassifier"]
