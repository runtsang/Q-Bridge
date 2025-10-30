"""Hybrid regression model combining classical CNN, LSTM and attention layers."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- #
# Dataset utilities
# --------------------------------------------------------------------------- #

def generate_regression_data(num_features: int, samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a synthetic regression dataset with sinusoidal target."""
    x = torch.rand(samples, num_features) * 2 - 1  # uniform in [-1, 1]
    angles = x.sum(dim=1)
    y = torch.sin(angles) + 0.1 * torch.cos(2 * angles)
    return x.float(), y.float()

class RegressionDataset(Dataset):
    """Torch dataset that returns feature vectors and scalar targets."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_regression_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {"features": self.features[idx], "target": self.labels[idx]}

# --------------------------------------------------------------------------- #
# Classical attention helper (adapted from SelfAttention.py)
# --------------------------------------------------------------------------- #

class ClassicalSelfAttention(nn.Module):
    """Simple dot‑product attention over a sequence of vectors."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.w_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_v = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (seq_len, batch, embed_dim)
        Returns:
            context: (seq_len, batch, embed_dim)
        """
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.embed_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v)

# --------------------------------------------------------------------------- #
# Classical hybrid regression model
# --------------------------------------------------------------------------- #

class HybridRegressionModel(nn.Module):
    """
    Classical network: 1D CNN ➜ LSTM ➜ Attention ➜ Linear head.
    Designed to mirror the quantum architecture for direct comparison.
    """
    def __init__(self, num_features: int, hidden_dim: int = 32, lstm_layers: int = 1):
        super().__init__()
        # Convolutional front‑end
        self.feature_map = nn.Sequential(
            nn.Linear(num_features, 16), nn.Tanh()
        )
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())

        # Recurrent core
        self.lstm = nn.LSTM(
            input_size=4, hidden_size=hidden_dim,
            num_layers=lstm_layers, batch_first=True
        )

        # Attention
        self.attention = ClassicalSelfAttention(embed_dim=hidden_dim)

        # Output head
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, features) or (batch, features) for single‑step regression.
        Returns:
            predictions: (batch, 1)
        """
        # Ensure 3‑D input
        if x.ndim == 2:
            x = x.unsqueeze(1)  # (batch, 1, features)

        seq_len = x.size(1)

        # Flatten batch‑seq dimension for CNN
        x_flat = x.reshape(-1, x.size(-1))
        x = self.feature_map(x_flat)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)

        # Reshape back to (batch, seq_len, features)
        x = x.reshape(-1, seq_len, x.size(-1))

        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)

        # Attention over sequence dimension
        attn_out = self.attention(lstm_out.transpose(0, 1)).transpose(0, 1)  # (batch, seq_len, hidden_dim)

        # Pool over sequence (mean) and head
        pooled = attn_out.mean(dim=1)
        out = self.head(pooled).squeeze(-1)
        return out

__all__ = ["RegressionDataset", "HybridRegressionModel", "generate_regression_data"]
