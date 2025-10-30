"""Unified hybrid regression model with classical LSTM for time‑series."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- #
# 1) Dataset helpers – identical to the Classical seed, but with optional
#    support for batched quantum feature vectors.
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Return `(X, y)` where `X∈R^{samples×num_features}` and `y` is a noisy sin‑like target."""
    rng = np.random.default_rng()
    X = rng.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = X.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return X, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset that yields a dict with state tensors and targets."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


# --------------------------------------------------------------------------- #
# 2) Classical dense feature extractor – a lightweight feed‑forward net.
# --------------------------------------------------------------------------- #
class DenseEncoder(nn.Module):
    """Simple 3‑layer FFN that mimics the original `QModel` linear stack."""
    def __init__(self, in_features: int, hidden: int = 32, out_features: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------------- #
# 3) Classical LSTM – can be swapped with the quantum variant.
# --------------------------------------------------------------------------- #
class ClassicalQLSTM(nn.Module):
    """Drop‑in replacement using classical linear gates."""
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int = 1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        return out


# --------------------------------------------------------------------------- #
# 4) Unified regression model – classical backbone.
# --------------------------------------------------------------------------- #
class UnifiedRegressionQLSTM(nn.Module):
    """
    Hybrid regression model that combines a dense encoder with a classical
    LSTM for time‑series regression.  The model can be trained end‑to‑end
    on batched data.  It mirrors the structure of the original quantum
    regression example while extending it to sequential inputs.
    """
    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        output_dim: int = 1,
        lstm_layers: int = 1,
    ) -> None:
        super().__init__()
        self.encoder = DenseEncoder(num_features, out_features=hidden_dim)
        self.lstm = ClassicalQLSTM(hidden_dim, hidden_dim, lstm_layers)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, num_features)
        Returns
        -------
        torch.Tensor
            Predicted values of shape (batch, seq_len, output_dim)
        """
        encoded = self.encoder(x)  # (batch, seq_len, hidden_dim)
        lstm_out = self.lstm(encoded)  # (batch, seq_len, hidden_dim)
        preds = self.head(lstm_out)  # (batch, seq_len, output_dim)
        return preds

__all__ = ["UnifiedRegressionQLSTM", "RegressionDataset", "generate_superposition_data"]
