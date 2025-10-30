"""Classical attention‑classifier hybrid.

This module builds on the SelfAttention and QuantumClassifierModel seeds.
It defines a `CombinedAttentionClassifier` that first applies a classical
self‑attention block and then feeds the resulting representation into a
compact feed‑forward classifier.  The design mirrors the quantum
counterpart while remaining fully classical (NumPy / PyTorch).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Iterable

# Import the classical classifier factory from the seed.
from build_classifier_circuit import build_classifier_circuit

class CombinedAttentionClassifier:
    """Classical self‑attention followed by a lightweight classifier."""

    def __init__(self, embed_dim: int, classifier_depth: int):
        """
        Parameters
        ----------
        embed_dim : int
            Dimensionality of the input features and the attention embedding.
        classifier_depth : int
            Number of hidden layers in the feed‑forward classifier.
        """
        self.embed_dim = embed_dim
        self.attention = self._build_attention()
        self.classifier, _, _, _ = build_classifier_circuit(
            num_features=embed_dim, depth=classifier_depth
        )
        self.classifier.eval()

    def _build_attention(self) -> nn.Module:
        """Simple multi‑head attention implementation."""
        # For brevity we use a single head; the structure mirrors the
        # seed's SelfAttention but with explicit weight matrices.
        return nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim, bias=False),
            nn.Linear(self.embed_dim, self.embed_dim, bias=False),
            nn.Linear(self.embed_dim, self.embed_dim, bias=False),
            nn.Softmax(dim=-1),
        )

    def run(
        self,
        inputs: np.ndarray,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> np.ndarray:
        """
        Forward pass through attention and classifier.

        Parameters
        ----------
        inputs : np.ndarray
            Input data of shape (batch, features).
        rotation_params : np.ndarray
            Parameters for the query projection.
        entangle_params : np.ndarray
            Parameters for the key projection.

        Returns
        -------
        np.ndarray
            Classifier logits of shape (batch, 2).
        """
        # Attention
        query = torch.as_tensor(
            inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32
        )
        key = torch.as_tensor(
            inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32
        )
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        attended = scores @ torch.as_tensor(inputs, dtype=torch.float32)

        # Classification
        logits = self.classifier(attended).detach().numpy()
        return logits

__all__ = ["CombinedAttentionClassifier"]
