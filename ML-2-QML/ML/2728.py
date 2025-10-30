"""Hybrid LSTM with optional regression head, classical implementation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, Optional

class QLSTM(nn.Module):
    """Classical LSTM cell with optional regression head."""
    def __init__(self, input_dim: int, hidden_dim: int, tagset_size: Optional[int] = None, n_qubits: int = 0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        if tagset_size is not None:
            self.tag_head = nn.Linear(hidden_dim, tagset_size)
        self.regression_head = nn.Linear(hidden_dim, 1)

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
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

    def forward_tagging(self, inputs: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.forward(inputs)
        logits = self.tag_head(lstm_out)
        return F.log_softmax(logits, dim=1)

    def forward_regression(self, inputs: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.forward(inputs)
        return self.regression_head(lstm_out.squeeze(-1))

    def _init_states(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return torch.zeros(batch_size, self.hidden_dim, device=device), torch.zeros(batch_size, self.hidden_dim, device=device)

class LSTMTagger(nn.Module):
    """Sequence tagging model using QLSTM."""
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int, n_qubits: int = 0):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM(embedding_dim, hidden_dim, tagset_size=tagset_size, n_qubits=n_qubits)
    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        return self.lstm.forward_tagging(embeds.view(len(sentence), 1, -1))

class RegressionDataset(Dataset):
    """Classical regression dataset."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)
    def __len__(self) -> int:
        return len(self.features)
    def __getitem__(self, idx: int) -> dict:
        return {"states": torch.tensor(self.features[idx], dtype=torch.float32),
                "target": torch.tensor(self.labels[idx], dtype=torch.float32)}

def generate_superposition_data(num_features: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionModel(nn.Module):
    """Simple feedforward regression model."""
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
        return self.net(state_batch.to(torch.float32)).squeeze(-1)

__all__ = ["QLSTM", "LSTMTagger", "RegressionDataset", "RegressionModel", "generate_superposition_data"]
