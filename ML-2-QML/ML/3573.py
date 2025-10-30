"""Hybrid classical model combining CNN, self‑attention, and dense projection.

The model mirrors the original Quantum‑NAT but augments the feature extractor
with a trainable classical self‑attention block.  Rotation and entanglement
parameters are learned as part of the module, allowing the network to
adapt the attention weights during training.

The architecture:
1. 2‑layer ConvNet → AvgPool → 16‑dim feature vector
2. Classical self‑attention on the 16‑dim vector
3. 2‑layer FC → BatchNorm → 4‑dim output
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassicalSelfAttention(nn.Module):
    """
    Lightweight self‑attention block that operates on a 1‑D feature vector.
    Rotation and entanglement parameters are trainable tensors that
    determine query/key projections.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(
        self,
        inputs: torch.Tensor,
        rotation_params: torch.Tensor,
        entangle_params: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            inputs: [batch, embed_dim]
            rotation_params: [embed_dim * 3]  (used to compute query)
            entangle_params: [embed_dim * 3]  (used to compute key)
        Returns:
            Attention‑weighted feature vector of shape [batch, embed_dim]
        """
        query = torch.matmul(inputs, rotation_params.reshape(self.embed_dim, -1))
        key = torch.matmul(inputs, entangle_params.reshape(self.embed_dim, -1))
        scores = torch.softmax(query @ key.T / math.sqrt(self.embed_dim), dim=-1)
        return scores @ inputs


class QuantumNATHybrid(nn.Module):
    """
    Classical counterpart of the hybrid Quantum‑NAT model.
    The forward pass applies the CNN, a self‑attention module, and a final
    fully‑connected head.  All parameters are trainable via PyTorch.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Reduce image to a 16‑dim vector via 6×6 average pooling
        self.pool = nn.AvgPool2d(kernel_size=6)
        self.attention = ClassicalSelfAttention(embed_dim=16)
        # Trainable attention parameters
        self.rotation_params = nn.Parameter(torch.randn(16 * 3))
        self.entangle_params = nn.Parameter(torch.randn(16 * 3))
        # Fully‑connected head
        self.fc = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, 1, H, W]  (e.g., 28×28 MNIST)
        Returns:
            [batch, 4]  normalized logits
        """
        # Feature extraction
        feat = self.features(x)          # shape: [batch, 16, 7, 7]
        pooled = self.pool(feat)         # shape: [batch, 16, 1, 1]
        pooled = pooled.view(x.shape[0], -1)  # shape: [batch, 16]

        # Classical self‑attention
        attn_out = self.attention(
            pooled, self.rotation_params, self.entangle_params
        )  # shape: [batch, 16]

        # Fully‑connected head
        out = self.fc(attn_out)          # shape: [batch, 4]
        return self.norm(out)


__all__ = ["QuantumNATHybrid"]
