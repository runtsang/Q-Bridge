"""Hybrid classical LSTM model for sequence tagging and regression.

The module exposes a single ``HybridQLSTM`` class that can be used for
both tasks.  It builds on the original ``QLSTM`` implementation but
adds a lightweight regression head.  The design keeps the API
compatible with the anchor ``QLSTM.py`` while extending the model
capabilities.

The class also includes utilities for generating a synthetic
superposition dataset and a corresponding ``RegressionDataset``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# --------------------------------------------------------------------------- #
#  Data utilities – classical superposition dataset
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic regression dataset where each sample is a vector
    of real numbers.  The target is a sinusoidal function of the
    element‑wise sum, mimicking a superposition angle.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input vectors.
    samples : int
        Number of samples to generate.

    Returns
    -------
    x : np.ndarray, shape (samples, num_features)
        Feature matrix.
    y : np.ndarray, shape (samples,)
        Regression targets.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """
    Torch ``Dataset`` wrapping the synthetic superposition data.
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


# --------------------------------------------------------------------------- #
#  Hybrid model – classical
# --------------------------------------------------------------------------- #
class HybridQLSTM(nn.Module):
    """
    Classical hybrid model that performs sequence tagging with an LSTM
    and simultaneously predicts a scalar regression target.

    Parameters
    ----------
    embedding_dim : int
        Size of word embeddings.
    hidden_dim : int
        Hidden size of the LSTM.
    vocab_size : int
        Size of the vocabulary.
    tagset_size : int
        Number of output tags.
    num_features : int
        Dimensionality of the regression input.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        num_features: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.tag_head = nn.Linear(hidden_dim, tagset_size)

        # Regression head – a small feed‑forward network
        self.reg_head = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(
        self,
        sentence: torch.Tensor,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        sentence : torch.Tensor, shape (batch, seq_len)
            Packed token indices.
        features : torch.Tensor, shape (batch, num_features)
            Regression input.

        Returns
        -------
        tag_loss : torch.Tensor
            Log‑softmax of tag logits, shape (batch, seq_len, tagset_size).
        reg_out : torch.Tensor
            Regression predictions, shape (batch,).
        """
        embeds = self.embedding(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.tag_head(lstm_out)
        tag_loss = F.log_softmax(tag_logits, dim=-1)
        reg_out = self.reg_head(features).squeeze(-1)
        return tag_loss, reg_out


__all__ = ["HybridQLSTM", "RegressionDataset", "generate_superposition_data"]
