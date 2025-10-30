"""Unified classical regression model that fuses classical and quantum‑style modules."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- #
# Data generation – hybrid‑style superposition data
# --------------------------------------------------------------------------- #
def generate_superposition_data(
    num_features: int,
    samples: int,
    *,
    noise: float = 0.1,
    phase_offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Produce a dataset that mimics the quantum superposition used in the seed
    while keeping the data purely classical.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    theta = x.sum(axis=1)
    y = np.sin(theta) + noise * np.cos(2 * theta + phase_offset)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset compatible with the original quantum regression example."""
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
# Classical self‑attention helper
# --------------------------------------------------------------------------- #
class ClassicalSelfAttention(nn.Module):
    """Self‑attention block that mimics the quantum circuit interface."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        scores = torch.softmax(Q @ K.transpose(-2, -1) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ V

# --------------------------------------------------------------------------- #
# Classical LSTM helper
# --------------------------------------------------------------------------- #
class ClassicalLSTMBlock(nn.Module):
    """Simple LSTM wrapper that returns the last hidden state."""
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, feature)
        _, (h_n, _) = self.lstm(x)
        return h_n.squeeze(0)  # (batch, hidden_dim)

# --------------------------------------------------------------------------- #
# Classical backbone
# --------------------------------------------------------------------------- #
class ClassicalBackbone(nn.Module):
    """Small fully‑connected network with residual skip."""
    def __init__(self, num_features: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.skip = nn.Linear(num_features, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.skip(x)

# --------------------------------------------------------------------------- #
# Unified regression model
# --------------------------------------------------------------------------- #
class UnifiedQuantumRegressor(nn.Module):
    """
    A regression model that can operate in three modes:
        * ``quantum`` – expects a user‑supplied quantum module that implements
          ``forward(states)`` and returns a feature tensor.
        * ``attention`` – uses a classical self‑attention block.
        * ``lstm`` – uses a classical LSTM block.
    The final prediction head is a linear layer mapping the hybrid features to a scalar.
    """
    def __init__(
        self,
        num_features: int,
        num_wires: int,
        mode: str = "attention",
        *,
        quantum_module: nn.Module | None = None,
        attention_embed_dim: int = 32,
        lstm_hidden_dim: int = 64,
    ):
        super().__init__()
        self.mode = mode
        self.backbone = ClassicalBackbone(num_features)

        if mode == "quantum":
            if quantum_module is None:
                raise ValueError("quantum_module must be provided when mode='quantum'")
            self.hybrid = quantum_module
        elif mode == "attention":
            self.hybrid = ClassicalSelfAttention(attention_embed_dim)
        elif mode == "lstm":
            self.hybrid = ClassicalLSTMBlock(num_features, lstm_hidden_dim)
        else:
            raise ValueError(f"Unsupported mode {mode!r}")

        # Head: map hybrid features to a scalar
        if mode == "quantum":
            hybrid_out_dim = quantum_module.out_dim  # type: ignore[assignment]
        elif mode == "attention":
            hybrid_out_dim = attention_embed_dim
        else:  # lstm
            hybrid_out_dim = lstm_hidden_dim

        self.head = nn.Linear(hybrid_out_dim, 1)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        states : torch.Tensor
            Shape (batch, num_features).

        Returns
        -------
        torch.Tensor
            Shape (batch,).
        """
        x = self.backbone(states)
        # For LSTM, add a dummy sequence dimension
        if self.mode == "lstm":
            x = x.unsqueeze(1)
        features = self.hybrid(x)
        return self.head(features).squeeze(-1)

__all__ = [
    "generate_superposition_data",
    "RegressionDataset",
    "ClassicalSelfAttention",
    "ClassicalLSTMBlock",
    "ClassicalBackbone",
    "UnifiedQuantumRegressor",
]
