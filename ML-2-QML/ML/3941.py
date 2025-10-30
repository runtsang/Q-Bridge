"""Hybrid regression and sequence model with fully classical backbone.

The ``HybridRegressionQLSTM`` class is a drop‑in replacement for the
classic ``QModel`` / ``QLSTM`` models.  It stitches together three
sub‑modules:
    1. a dense feed‑forward network that handles the feature dimension,
    2. a classical LSTM that processes the sequence of latent vectors,
    3. a regression head that produces the final scalar output.

The design follows the structure of the two seed sources but extends
their capabilities: the first linear stage learns a richer feature
representation, the second stage captures temporal dependencies, and
the final head projects the hidden state to a scalar.  The model is
fully compatible with the dataset generators
(`generate_superposition_data` and ``RegressionDataset``) and exposes
the torch‑friendly API used in the reference seeds.

Author: <your name>
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Tuple, Optional

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """
    Dataset that returns a feature vector and its regression target.
    """
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

class ClassicalQLSTM(nn.Module):
    """
    Pure PyTorch implementation mirroring the quantum LSTM interfaces.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0):
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
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

class LSTMTagger(nn.Module):
    """
    Sequence tagging model that uses a classical LSTM.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            raise ValueError("Classical implementation cannot use quantum gates; set n_qubits=0.")
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=-1)

class HybridRegressionQLSTM(nn.Module):
    """
    Classical hybrid regression + LSTM model.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input feature vectors.
    hidden_dim : int
        Size of the hidden state in the LSTM.
    n_qubits : int, default 0
        If >0, the model is intended to be used with the quantum version
        (the classical implementation ignores this flag but keeps it for API
        consistency).
    """
    def __init__(self, num_features: int, hidden_dim: int, n_qubits: int = 0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, hidden_dim),
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, seq_len, num_features)

        Returns
        -------
        torch.Tensor
            Regression output of shape (batch, seq_len)
        """
        batch, seq_len, _ = x.shape
        x_flat = x.view(batch * seq_len, -1)
        encoded = self.encoder(x_flat)
        encoded = encoded.view(batch, seq_len, -1)
        lstm_out, _ = self.lstm(encoded)
        out = self.regressor(lstm_out).squeeze(-1)
        return out

__all__ = [
    "generate_superposition_data",
    "RegressionDataset",
    "ClassicalQLSTM",
    "LSTMTagger",
    "HybridRegressionQLSTM",
]
