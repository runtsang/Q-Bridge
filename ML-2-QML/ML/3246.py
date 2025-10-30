"""
HybridSamplerQNN: Classical hybrid model combining QCNN‑style feature extraction and a sampler head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridSamplerQNNModel(nn.Module):
    """
    A PyTorch model that mimics a quantum convolutional neural network (QCNN) using
    fully connected layers, followed by a lightweight sampler head that outputs a
    probability distribution over two classes.
    """

    def __init__(self) -> None:
        super().__init__()

        # Feature extractor: QCNN‑style stack of linear layers with Tanh activations
        self.feature_map = nn.Sequential(
            nn.Linear(8, 16),
            nn.Tanh(),
        )
        self.conv1 = nn.Sequential(
            nn.Linear(16, 16),
            nn.Tanh(),
        )
        self.pool1 = nn.Sequential(
            nn.Linear(16, 12),
            nn.Tanh(),
        )
        self.conv2 = nn.Sequential(
            nn.Linear(12, 8),
            nn.Tanh(),
        )
        self.pool2 = nn.Sequential(
            nn.Linear(8, 4),
            nn.Tanh(),
        )
        self.conv3 = nn.Sequential(
            nn.Linear(4, 4),
            nn.Tanh(),
        )

        # Classification head: maps the final feature vector to logits
        self.head = nn.Linear(4, 1)

        # Sampler head: projects the QCNN output to a 2‑class probability vector
        self.sampler_head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 8).

        Returns
        -------
        torch.Tensor
            Softmax probabilities over two classes.
        """
        # QCNN‑style feature extraction
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)

        # Optional classification head
        logits = self.head(x)

        # Sampler head to produce a 2‑class distribution
        sampler_logits = self.sampler_head(x)
        probs = F.softmax(sampler_logits, dim=-1)
        return probs


def HybridSamplerQNN() -> HybridSamplerQNNModel:
    """
    Factory that returns a configured instance of :class:`HybridSamplerQNNModel`.
    """
    return HybridSamplerQNNModel()


__all__ = ["HybridSamplerQNN", "HybridSamplerQNNModel"]
