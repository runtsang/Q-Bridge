"""Hybrid self‑attention model with classical attention and regression.

This module combines the classical self‑attention block from the original
SelfAttention seed with a lightweight regression head.  The class exposes
the same API as the original seed but adds a training loop and a regression
dataset that can be used for downstream experiments.

Typical usage::

    from SelfAttention__gen063 import SelfAttention
    model = SelfAttention()
    out = model.run_classical_attention(rot, ent, inp)
    model.train_regression(dataset, epochs=20)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# --------------------------------------------------------------------------- #
#  Classical self‑attention helper
# --------------------------------------------------------------------------- #
class ClassicalSelfAttention:
    """Pure‑Python / PyTorch implementation of a single attention block."""
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        # Map parameters to linear layers
        W_q = torch.as_tensor(rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        W_k = torch.as_tensor(entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        # Compute Q, K, V
        Q = torch.as_tensor(inputs @ W_q, dtype=torch.float32)
        K = torch.as_tensor(inputs @ W_k, dtype=torch.float32)
        V = torch.as_tensor(inputs, dtype=torch.float32)
        # Attention scores
        scores = F.softmax(Q @ K.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ V).numpy()

# --------------------------------------------------------------------------- #
#  Classical regression head
# --------------------------------------------------------------------------- #
class ClassicalRegressionModel(nn.Module):
    """Simple feed‑forward network for regression on attention outputs."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

# --------------------------------------------------------------------------- #
#  Data utilities
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data resembling a superposition of basis states."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset that returns a feature vector and a target value."""
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
#  Hybrid model combining attention and regression
# --------------------------------------------------------------------------- #
class HybridSelfAttentionModel:
    """Public API that mimics the original SelfAttention seed but adds a
    regression head and training utilities."""
    def __init__(self, embed_dim: int = 4, n_features: int = 4):
        self.embed_dim = embed_dim
        self.n_features = n_features
        self.attention = ClassicalSelfAttention(embed_dim)
        self.regressor = ClassicalRegressionModel(input_dim=n_features)

    # --------------------------------------------------------------------- #
    #  Classical attention
    # --------------------------------------------------------------------- #
    def run_classical_attention(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """Forward pass through the classical attention block."""
        return self.attention.run(rotation_params, entangle_params, inputs)

    # --------------------------------------------------------------------- #
    #  Regression training
    # --------------------------------------------------------------------- #
    def train_regression(
        self,
        dataset: Dataset,
        epochs: int = 10,
        lr: float = 1e-3,
        batch_size: int = 32,
        device: str | torch.device = "cpu",
    ) -> None:
        """Train the regression head on the supplied dataset."""
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.regressor.to(device)
        optimizer = torch.optim.Adam(self.regressor.parameters(), lr=lr)
        criterion = nn.MSELoss()
        self.regressor.train()
        for _ in range(epochs):
            for batch in loader:
                states = batch["states"].to(device)
                target = batch["target"].to(device)
                optimizer.zero_grad()
                pred = self.regressor(states)
                loss = criterion(pred, target)
                loss.backward()
                optimizer.step()

    # --------------------------------------------------------------------- #
    #  Evaluation
    # --------------------------------------------------------------------- #
    def evaluate(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        device: str | torch.device = "cpu",
    ) -> float:
        """Return mean‑squared error on the dataset."""
        loader = DataLoader(dataset, batch_size=batch_size)
        self.regressor.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for batch in loader:
                states = batch["states"].to(device)
                target = batch["target"].to(device)
                pred = self.regressor(states)
                total += ((pred - target) ** 2).sum().item()
                n += target.numel()
        return total / n

# --------------------------------------------------------------------------- #
#  Factory function that preserves the original API
# --------------------------------------------------------------------------- #
def SelfAttention() -> HybridSelfAttentionModel:
    """Return a classical hybrid self‑attention model."""
    return HybridSelfAttentionModel()

__all__ = ["HybridSelfAttentionModel", "SelfAttention"]
