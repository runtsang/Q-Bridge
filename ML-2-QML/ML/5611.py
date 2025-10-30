"""Hybrid classical regression model integrating LSTM, transformer, kernel, and quantum‑like features."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# Data generation utilities
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# Classical feature extractor
class RandomFeatureLayer(nn.Module):
    """Random Fourier feature approximation of a quantum kernel."""
    def __init__(self, input_dim: int, output_dim: int, gamma: float = 1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(output_dim, input_dim) * np.sqrt(2 * gamma), requires_grad=False)
        self.b = nn.Parameter(torch.randn(output_dim) * 2 * np.pi, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.cos(x @ self.W.t() + self.b)
        return z

# Positional encoding used by transformer
class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-np.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]

# Hybrid regression model
class QModel(nn.Module):
    """Hybrid classical regression model that optionally uses LSTM, transformer, kernel, and quantum‑like features."""
    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 32,
        use_lstm: bool = False,
        use_transformer: bool = False,
        use_kernel: bool = False,
        use_quantum: bool = False,
        kernel_dim: int = 64,
        quantum_dim: int = 64,
    ):
        super().__init__()
        self.use_lstm = use_lstm
        self.use_transformer = use_transformer
        self.use_kernel = use_kernel
        self.use_quantum = use_quantum

        if use_lstm:
            self.lstm = nn.LSTM(num_features, hidden_dim, batch_first=True)
        else:
            self.lstm = None

        if use_transformer:
            self.pos_encoder = PositionalEncoder(num_features)
            self.transformer = nn.Transformer(
                d_model=num_features,
                nhead=4,
                num_encoder_layers=2,
                batch_first=True,
            )
        else:
            self.transformer = None

        if use_kernel:
            self.kernel_layer = RandomFeatureLayer(num_features, kernel_dim, gamma=1.0)
        else:
            self.kernel_layer = None

        if use_quantum:
            self.quantum_layer = RandomFeatureLayer(num_features, quantum_dim, gamma=0.5)
        else:
            self.quantum_layer = None

        # Final MLP head
        input_dim = num_features
        if use_lstm:
            input_dim = hidden_dim
        if use_transformer:
            input_dim = num_features
        if use_kernel:
            input_dim += kernel_dim
        if use_quantum:
            input_dim += quantum_dim

        self.head = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, features) for sequence models
            or (batch, features) for pointwise regression.
        """
        if self.use_lstm:
            out, _ = self.lstm(x)
            x = out[:, -1, :]
        if self.use_transformer:
            x = self.pos_encoder(x)
            x = self.transformer(x)
            x = x.mean(dim=1)
        features = [x]
        if self.use_kernel:
            features.append(self.kernel_layer(x))
        if self.use_quantum:
            features.append(self.quantum_layer(x))
        x = torch.cat(features, dim=-1)
        return self.head(x).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
