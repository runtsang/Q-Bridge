"""Hybrid classical LSTM implementation with optional quantum‑inspired gates and regression support."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

__all__ = ["QLSTM", "LSTMTagger", "RegressionDataset", "RegressionModel"]


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Classical analogue of the quantum superposition dataset.
    Generates features uniformly in [-1, 1] and labels with a sinusoidal pattern.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset for the classical regression task."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class RegressionModel(nn.Module):
    """Simple feed‑forward network mimicking the quantum regression head."""

    def __init__(self, num_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(state_batch).squeeze(-1)


class QLSTM(nn.Module):
    """
    Classical LSTM cell that optionally uses quantum‑inspired MLP gates.
    The `n_qubits` flag is kept for API compatibility; it does not trigger
    quantum operations in this module.
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0, gate_hidden_dim: int = 32) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits  # retained for API consistency

        # Linear projections into a gate representation space
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_hidden_dim)

        # Mapping from gate representation to hidden dimension
        self.forget_gate = nn.Linear(gate_hidden_dim, hidden_dim)
        self.input_gate = nn.Linear(gate_hidden_dim, hidden_dim)
        self.update_gate = nn.Linear(gate_hidden_dim, hidden_dim)
        self.output_gate = nn.Linear(gate_hidden_dim, hidden_dim)

        # Optional noise injection to emulate quantum stochasticity
        self.noise = nn.Dropout(p=0.1)

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_gate(F.relu(self.forget_linear(combined))))
            i = torch.sigmoid(self.input_gate(F.relu(self.input_linear(combined))))
            g = torch.tanh(self.update_gate(F.relu(self.update_linear(combined))))
            o = torch.sigmoid(self.output_gate(F.relu(self.output_linear(combined))))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            hx = self.noise(hx)  # stochasticity
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
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )


class LSTMTagger(nn.Module):
    """
    Sequence tagging model that can switch between classical LSTM and the
    quantum‑inspired LSTM defined above.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)
