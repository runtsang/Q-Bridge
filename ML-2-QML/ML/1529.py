"""Advanced classical sampler network with skip‑connections and dropout."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdvancedSamplerQNN(nn.Module):
    """
    A deeper sampler network that mirrors the original architecture but
    incorporates residual connections, batch‑normalisation and dropout to
    improve generalisation on noisy data.

    The network processes a 2‑dimensional input and returns a probability
    distribution over two classes.  It can be used inside any
    variational‑learning pipeline that expects a torch.nn.Module.
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(hidden_dim, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass with a residual connection from the input to the
        classifier output.  The final softmax normalises the logits.
        """
        features = self.feature_extractor(inputs)
        # Residual connection: add a linear projection of the input
        residual = nn.functional.linear(inputs, torch.eye(inputs.size(-1), self.classifier.out_features, device=inputs.device))
        logits = self.classifier(features + residual)
        return F.softmax(logits, dim=-1)


__all__ = ["AdvancedSamplerQNN"]
