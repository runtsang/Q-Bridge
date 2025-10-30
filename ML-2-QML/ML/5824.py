"""Hybrid classical LSTM with optional regression head and dataset utilities.

This module extends the original QLSTM implementation by adding:
- A regression head that can be used for quantum-inspired regression experiments.
- Dataset utilities that generate superposition-like features for regression tasks.
- A flexible tagging model that can switch between classification and regression modes.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# ----------------------------------------------------------------------
# Data utilities
# ----------------------------------------------------------------------
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data that mimics the structure of quantum superposition states.
    The function is a classical analogue of the quantum generation used in the QML seed.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """
    Dataset for regression experiments using the synthetic superposition data.
    """
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

# ----------------------------------------------------------------------
# Core LSTM cell
# ----------------------------------------------------------------------
class QLSTM(nn.Module):
    """
    Classical LSTM cell that can be used as a drop‑in replacement for the quantum cell.
    The implementation is identical to the original QLSTM, but an optional
    regression head is added to demonstrate how the same cell can serve
    different downstream tasks.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits  # retained for API compatibility
        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

        # Regression head – maps hidden state to a scalar target
        self.regression_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
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

# ----------------------------------------------------------------------
# Tagging / regression model
# ----------------------------------------------------------------------
class LSTMTagger(nn.Module):
    """
    Sequence tagging model that can operate in three modes:
    - 'tagging'  : returns log‑softmax logits for each token
    -'regression': returns a scalar per token
    - 'both'      : returns both outputs
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        mode: str = "tagging",
    ) -> None:
        super().__init__()
        self.mode = mode
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden2reg = nn.Linear(hidden_dim, 1)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        if self.mode == "tagging":
            tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
            return F.log_softmax(tag_logits, dim=1)
        elif self.mode == "regression":
            reg_logits = self.hidden2reg(lstm_out.view(len(sentence), -1))
            return reg_logits.squeeze(-1)
        else:  # both
            tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
            reg_logits = self.hidden2reg(lstm_out.view(len(sentence), -1))
            return F.log_softmax(tag_logits, dim=1), reg_logits.squeeze(-1)

class QModel(nn.Module):
    """
    Simple feed‑forward regression network used as a baseline in the classical experiments.
    """
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

__all__ = ["QLSTM", "LSTMTagger", "RegressionDataset", "QModel", "generate_superposition_data"]
