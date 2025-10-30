from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QLSTMRegressor(nn.Module):
    """Classical LSTM regressor with optional quantum gate support.

    The model uses an nn.LSTM backbone and a linear output head.
    It can be used as a drop‑in replacement for the original QLSTM
    while keeping a pure classical back‑end.  The architecture
    mirrors the quantum version, enabling side‑by‑side experiments.
    """
    def __init__(self, input_dim: int, hidden_dim: int,
                 output_dim: int = 1, n_layers: int = 1,
                 bidirectional: bool = False):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers=n_layers,
                            batch_first=True,
                            bidirectional=bidirectional)
        self.head = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, feature)
        out, _ = self.lstm(x)
        # Use last time step for regression
        out = out[:, -1, :]
        return self.head(out)

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate real‑valued superposition features and a sinusoid target."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class SequenceRegressionDataset(torch.utils.data.Dataset):
    """Dataset of sequences of superposition states for regression."""
    def __init__(self, samples: int, seq_len: int, num_features: int):
        self.samples = samples
        self.seq_len = seq_len
        self.num_features = num_features
        self.features, self.labels = generate_superposition_data(num_features * seq_len, samples)
        # Reshape into sequences
        self.features = self.features.reshape(samples, seq_len, num_features)

    def __len__(self) -> int:  # type: ignore[override]
        return self.samples

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "sequence": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

__all__ = ["QLSTMRegressor", "generate_superposition_data", "SequenceRegressionDataset"]
