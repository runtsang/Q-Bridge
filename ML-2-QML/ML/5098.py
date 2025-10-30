"""
Hybrid classical estimator that optionally incorporates LSTM, convolutional,
quantum‑LSTM, quanvolution, and a regression head.

The class is intentionally *configurable*: the same object can be used
in a purely classical setting or with a quantum‑enhanced LSTM or quanvolution
layer, simply by passing the appropriate flags at construction time.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data in a superposition‑style
    (see the quantum example for the complex version).
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """
    Pure‑Python dataset mirroring the quantum regression example.
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


class ClassicalQLSTM(nn.Module):
    """
    Simple LSTM cell using only classical linear layers.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return out


class EstimatorQNNGen490(nn.Module):
    """
    Hybrid estimator that can operate in a number of modes:

    - **feed‑forward only**: a small fully‑connected network (default).
    - **LSTM**: classical or quantum, selectable via ``use_lstm`` and ``n_qubits``.
    - **convolution / quanvolution**: 2‑D patch extraction, selectable via
      ``use_quanvolution``.
    - **regression head**: linear or quantum (not implemented in the classical
      branch – see :class:`~qml.EstimatorQNNGen490` for the quantum head).
    """
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 8,
        n_qubits: int = 0,
        use_lstm: bool = False,
        use_quanvolution: bool = False,
        use_quantum_head: bool = False,
        regression_wires: int = 4,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.use_lstm = use_lstm
        self.use_quanvolution = use_quanvolution
        self.use_quantum_head = use_quantum_head

        # Base feed‑forward block
        self.ff = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        # Optional LSTM
        if self.use_lstm:
            if self.n_qubits > 0:
                # In the classical branch we only expose the interface;
                # the quantum implementation lives in :mod:`qml`.
                self.lstm = None
            else:
                self.lstm = ClassicalQLSTM(input_dim, hidden_dim)
        else:
            self.lstm = None

        # Optional convolution / quanvolution
        if self.use_quanvolution:
            self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
            self.conv_out_dim = 4 * 14 * 14
        else:
            self.conv = None
            self.conv_out_dim = input_dim

        # Regression head
        if self.use_quantum_head:
            # The quantum head is only available in the quantum module;
            # we provide a placeholder to keep the API consistent.
            self.head = nn.Linear(1, 1)
        else:
            self.head = nn.Linear(self.conv_out_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            If ``use_quanvolution`` is ``True`` the tensor must have shape
            ``(batch, 1, 28, 28)``.  Otherwise it should be
            ``(batch, seq_len, features)`` if ``use_lstm`` is ``True`` or
            ``(batch, features)`` otherwise.
        """
        # 1. Convolution / Quanvolution
        if self.conv is not None:
            features = self.conv(x)
            features = features.view(x.size(0), -1)
        else:
            if self.lstm is not None and x.dim() == 3:
                # Sequence data for LSTM
                features, _ = self.lstm(x)
                features = features[:, -1, :]  # use last hidden state
            else:
                features = x  # raw input

        # 2. Feed‑forward
        ff_out = self.ff(features)

        # 3. Head
        out = self.head(ff_out)

        return out.squeeze(-1)


__all__ = [
    "EstimatorQNNGen490",
    "RegressionDataset",
    "generate_superposition_data",
    "ClassicalQLSTM",
]
