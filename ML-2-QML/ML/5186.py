"""Hybrid LSTM with classical QCNN feature extractor and sampler‑modulated gates.

The module is a drop‑in replacement for the original `QLSTM` but adds a
quantum‑inspired feature extractor (`QCNNModel`) and a lightweight
sampler network (`SamplerQNN`) that modulates the LSTM gates.  The
architecture is fully classical and can be trained with standard
PyTorch optimisers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

__all__ = [
    "HybridQLSTM",
    "HybridLSTM",
    "SamplerQNN",
    "QCNNModel",
    "RegressionDataset",
    "QModel",
]


class QCNNModel(nn.Module):
    """Classical surrogate of a quantum convolutional network.

    The structure mirrors the original QCNN while replacing each quantum
    block with a learnable linear layer followed by a tanh non‑linearity.
    The depth and fan‑out are preserved to keep the same receptive field
    while remaining fully classical.
    """

    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))


class SamplerQNN(nn.Module):
    """Small neural sampler that produces a per‑sample gate modulation.

    The network outputs a scalar in the unit interval that is used to
    element‑wise scale the classical LSTM gates.  It is intentionally
    shallow to keep the overhead negligible.
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 1),  # one scalar per sample
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs shape: (batch, features)
        return torch.sigmoid(self.net(inputs))  # (batch, 1)


class HybridLSTM(nn.Module):
    """Classical LSTM whose gates are modulated by :class:`SamplerQNN`."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.forget = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.sampler = SamplerQNN()

    def _init_states(self, inputs: torch.Tensor):
        batch_size = inputs.size(0)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hx, cx = self._init_states(inputs)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            mod = self.sampler(combined).unsqueeze(-1)  # (batch, 1) -> (batch, 1)
            f = torch.sigmoid(self.forget(combined)) * mod
            i = torch.sigmoid(self.input(combined)) * mod
            g = torch.tanh(self.update(combined))
            o = torch.sigmoid(self.output(combined)) * mod
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)


class HybridQLSTM(nn.Module):
    """Hybrid tagger that integrates QCNN, Sampler‑modulated LSTM and linear head."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,  # placeholder for compatibility
    ) -> None:
        super().__init__()
        self.qcnn = QCNNModel()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # The QCNN outputs 8 features; use this as the LSTM input dimension
        self.lstm = HybridLSTM(self.qcnn.feature_map[0].in_features, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        # QCNN operates on the embedding dimension; we flatten batch
        qcnn_out = self.qcnn(embeds)
        lstm_out, _ = self.lstm(qcnn_out)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=1)


class RegressionDataset(Dataset):
    """Dataset that generates superposition‑style samples for regression."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = self._generate(samples, num_features)

    @staticmethod
    def _generate(samples: int, num_features: int):
        x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
        angles = x.sum(axis=1)
        y = np.sin(angles) + 0.1 * np.cos(2 * angles)
        return x, y.astype(np.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class QModel(nn.Module):
    """Simple fully‑connected regression head."""

    def __init__(self, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        return self.net(state_batch).squeeze(-1)
