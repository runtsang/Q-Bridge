"""Hybrid classical model combining CNN, self‑attention and a regression head.

This module implements the `HybridNAT` class, a pure‑PyTorch
implementation that mirrors the structure of the original
`QuantumNAT` example but augments it with a classical
self‑attention block (inspired by the QML SelfAttention helper)
and a lightweight FastEstimator wrapper for rapid evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- Self‑attention helper ----------------------------------------------------
class ClassicalSelfAttention:
    """Simple dot‑product self‑attention operating on 1‑D feature vectors."""
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim
        # Learnable projection matrices
        self.W_q = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.W_k = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.W_v = nn.Parameter(torch.randn(embed_dim, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, embed_dim).

        Returns
        -------
        torch.Tensor
            Attention‑weighted representation of shape (batch, seq_len, embed_dim).
        """
        q = torch.matmul(x, self.W_q)
        k = torch.matmul(x, self.W_k)
        v = torch.matmul(x, self.W_v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.embed_dim)
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, v)


# --- FastEstimator utilities --------------------------------------------------
class FastEstimator:
    """
    Lightweight estimator that evaluates a PyTorch model for a batch of
    parameter sets and a list of scalar observables (functions).
    """
    def __init__(self, model: nn.Module):
        self.model = model

    def evaluate(
        self,
        observables: list[callable],
        parameter_sets: list[list[float]]
    ) -> list[list[float]]:
        self.model.eval()
        results = []
        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.as_tensor(params, dtype=torch.float32).unsqueeze(0)
                outputs = self.model(inputs)
                row = [float(obs(outputs).mean().cpu()) for obs in observables]
                results.append(row)
        return results


# --- Hybrid model -------------------------------------------------------------
class HybridNAT(nn.Module):
    """
    Classical hybrid model that processes 2‑D images with a CNN,
    applies a self‑attention block, and projects to 4‑dimensional outputs.
    """
    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 16,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Self‑attention on flattened spatial features
        self.attn = ClassicalSelfAttention(embed_dim=embed_dim)
        # Fully connected head
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        feat = self.features(x)                       # (bsz, 16, H', W')
        flat = feat.view(bsz, -1, 1)                  # (bsz, 16*H'*W', 1)
        # Apply attention along the feature dimension
        attn_out = self.attn(flat).squeeze(-1)        # (bsz, 16*H'*W')
        out = self.fc(attn_out)
        return self.norm(out)

    def estimator(self) -> FastEstimator:
        """Return a FastEstimator wrapper around this model."""
        return FastEstimator(self)

__all__ = ["HybridNAT"]
