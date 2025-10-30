"""Hybrid quantum‑classical regression model – classical component only."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic regression dataset with sinusoidal target.
    The data is intentionally simple to allow direct comparison between
    classical and quantum models.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.2 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """
    Dataset that returns a dictionary with ``states`` and ``target``.
    ``states`` are the raw feature vectors and ``target`` is the scalar label.
    """
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class ClassicalModel(nn.Module):
    """
    Feed‑forward network that maps raw features to a scalar prediction.
    Dropout is added for regularisation and the architecture is
    intentionally deeper than the original seed.
    """
    def __init__(self, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch.to(torch.float32)).squeeze(-1)


class QuantumRegressionHybrid(nn.Module):
    """
    Classical backbone that can optionally include a recurrent layer.
    The interface mirrors the quantum implementation so that the
    two modules can be swapped without changing downstream code.
    """
    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 64,
        use_lstm: bool = False,
        lstm_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.use_lstm = use_lstm
        if use_lstm:
            self.lstm = nn.LSTM(
                input_size=num_features,
                hidden_size=lstm_hidden_dim,
                batch_first=False,
            )
            self.lstm_head = nn.Linear(lstm_hidden_dim, 1)
        else:
            self.mlp = ClassicalModel(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs
            ``(batch, features)`` for a single‑step regression or
            ``(seq_len, batch, features)`` for a sequence.
        Returns
        -------
        torch.Tensor
            Prediction of shape ``(batch,)`` or ``(seq_len, batch)``.
        """
        if inputs.dim() == 3:  # sequence data
            outputs, _ = self.lstm(inputs)
            preds = self.lstm_head(outputs)
            return preds.squeeze(-1)
        else:  # single‑step data
            preds = self.mlp(inputs)
            return preds


__all__ = ["QuantumRegressionHybrid", "RegressionDataset", "generate_superposition_data"]
