"""Hybrid classical LSTM model with optional regression head.

This module defines a classical LSTM layer (QLSTM) that can be used for
sequence tagging or sequence regression.  The implementation is a
drop‑in replacement for the original QLSTM.py but with an additional
regression head that can be used for continuous targets.  The class
inherits from torch.nn.Module and is fully compatible with the
original tagger interface, while exposing a `regress` flag that
activates the regression head.

The code also contains utilities for generating a synthetic dataset
similar to the quantum regression example, but using classical
features.  The dataset can be used to train the regression head
without any quantum hardware.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- #
# Data utilities
# --------------------------------------------------------------------------- #

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data that mimics a quantum superposition but
    using only classical features.  The data is sampled from a
    distribution that is easy to model with a shallow neural network.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset that returns a pair (states, target)."""

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
# Classical LSTM cell
# --------------------------------------------------------------------------- #

class QLSTM(nn.Module):
    """Classical LSTM cell that mimics the interface of the quantum version."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:  # type: ignore[override]
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

# --------------------------------------------------------------------------- #
# Tagger / Regressor wrappers
# --------------------------------------------------------------------------- #

class LSTMTagger(nn.Module):
    """Sequence tagging model that uses :class:`QLSTM`."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)


class QLSTMRegressor(nn.Module):
    """
    Sequence regression model that processes a sequence of feature vectors
    with a classical LSTM and outputs a continuous target.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.lstm = QLSTM(input_dim, hidden_dim)
        self.regress_head = nn.Linear(hidden_dim, 1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        sequence : torch.Tensor
            Tensor of shape (seq_len, batch, input_dim).
        """
        lstm_out, _ = self.lstm(sequence)
        # Use the last hidden state for regression
        last_hidden = lstm_out[-1]
        return self.regress_head(last_hidden).squeeze(-1)

__all__ = ["QLSTM", "LSTMTagger", "QLSTMRegressor", "RegressionDataset", "generate_superposition_data"]
