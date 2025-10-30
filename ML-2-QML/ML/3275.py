"""Enhanced classical regression model with hybrid fully‑connected layer.

Implements a standard feed‑forward network augmented by a lightweight
parameterized quantum layer (implemented via a mock FCL) to provide
non‑linear feature transformations.  The model can be trained
entirely on a CPU, making it suitable as a baseline or as a
classical counterpart to the quantum version.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Iterable

# --------------------------------------------------------------------------- #
# Data generation utilities
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data using a superposition‑like pattern."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset yielding feature vectors and scalar targets."""
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
# Classical hybrid layer mimicking a quantum FCL
# --------------------------------------------------------------------------- #
def FCL() -> nn.Module:
    """Return a lightweight module that emulates a parameterised quantum
    fully‑connected layer.  The implementation is purely classical but
    preserves the interface used in the original quantum example.
    """
    class FullyConnectedLayer(nn.Module):
        def __init__(self, n_features: int = 1) -> None:
            super().__init__()
            # Simple linear mapping followed by a non‑linearity
            self.linear = nn.Linear(n_features, 1)

        def run(self, thetas: Iterable[float]) -> np.ndarray:
            """Compute a mock expectation value from a list of parameters."""
            values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
            expectation = torch.tanh(self.linear(values)).mean(dim=0)
            return expectation.detach().numpy()

    return FullyConnectedLayer()


# --------------------------------------------------------------------------- #
# Main model
# --------------------------------------------------------------------------- #
class QModel(nn.Module):
    """Classical regression model that combines a standard feed‑forward
    network with a mock quantum‑style fully‑connected layer.
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        # The mock quantum layer
        self.q_layer = FCL()
        # Final head
        self.head = nn.Linear(16, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        The output of the feature extractor is fed into the mock quantum
        layer; the resulting scalar is concatenated with the 16‑dimensional
        feature vector before the final linear head.
        """
        feats = self.feature_extractor(state_batch)
        # The mock quantum layer expects a list of parameters; we use the
        # mean of the features as a proxy for a parameter vector.
        q_out = self.q_layer.run(feats.mean(dim=1).tolist())
        q_tensor = torch.tensor(q_out, dtype=torch.float32, device=state_batch.device)
        # Concatenate and predict
        concat = torch.cat([feats, q_tensor], dim=1)
        return self.head(concat).squeeze(-1)


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
