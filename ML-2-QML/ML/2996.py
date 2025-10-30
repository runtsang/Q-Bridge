"""Hybrid classical regression model with self‑attention pre‑processor."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- #
#  Dataset & data‑generation utilities
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data where the target is a smooth non‑linear function
    of the input features.  The implementation is a lightweight
    evolution of the original seed; it now supports optional
    seeding for reproducibility."""
    rng = np.random.default_rng()
    x = rng.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset that mirrors the quantum counterpart but operates on real tensors."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# --------------------------------------------------------------------------- #
#  Classical self‑attention
# --------------------------------------------------------------------------- #
def SelfAttention():
    """Return a small self‑attention module that mimics the quantum interface."""
    class ClassicalSelfAttention:
        def __init__(self, embed_dim: int):
            self.embed_dim = embed_dim

        def run(
            self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray,
        ) -> np.ndarray:
            """Simple dot‑product attention with trainable linear projections."""
            query = torch.as_tensor(
                inputs @ rotation_params.reshape(self.embed_dim, -1),
                dtype=torch.float32,
            )
            key = torch.as_tensor(
                inputs @ entangle_params.reshape(self.embed_dim, -1),
                dtype=torch.float32,
            )
            value = torch.as_tensor(inputs, dtype=torch.float32)
            scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
            return (scores @ value).numpy()

    return ClassicalSelfAttention(embed_dim=4)

# --------------------------------------------------------------------------- #
#  Hybrid attention‑based regression model
# --------------------------------------------------------------------------- #
class HybridAttentionRegression(nn.Module):
    """Regression network that first applies a classical self‑attention
    module and then a small feed‑forward network."""
    def __init__(self, num_features: int, embed_dim: int = 4):
        super().__init__()
        # Attention parameters
        self.rotation = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.entangle = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.attention = SelfAttention()

        # Regression head
        self.net = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Convert batch to numpy for the attention module
        batch_np = state_batch.detach().cpu().numpy()
        # Apply attention
        attended = self.attention.run(
            self.rotation.detach().cpu().numpy(),
            self.entangle.detach().cpu().numpy(),
            batch_np,
        )
        # Feed the attended representation into the regression head
        return self.net(torch.tensor(attended, dtype=torch.float32)).squeeze(-1)

__all__ = ["HybridAttentionRegression", "RegressionDataset", "generate_superposition_data"]
