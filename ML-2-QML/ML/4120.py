"""Hybrid classical QCNN with optional quantum‑style LSTM integration."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Classical LSTM implementation from QLSTM.py
class ClassicalQLSTM(nn.Module):
    """Drop‑in classical replacement for the quantum LSTM gates."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

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
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

class QCNNHybrid(nn.Module):
    """Hybrid QCNN that combines classical convolution, pooling and a quantum‑style LSTM."""

    def __init__(
        self,
        input_dim: int = 8,
        lstm_hidden_dim: int = 16,
        n_qubits: int = 2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim

        # Feature extraction
        self.feature_map = nn.Sequential(nn.Linear(input_dim, 16), nn.Tanh())

        # Convolutional layers – simple fully‑connected mimicking quantum conv
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 12), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(8, 8), nn.Tanh())

        # Pooling layers – linear dimensionality reduction
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool3 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())

        # LSTM – classical LSTM mimicking quantum gates
        self.lstm = ClassicalQLSTM(4, lstm_hidden_dim, n_qubits=n_qubits)

        self.head = nn.Linear(lstm_hidden_dim, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(inputs)

        # Sequential conv → pool
        x1 = self.conv1(x)
        x1 = self.pool1(x1)
        x2 = self.conv2(x1)
        x2 = self.pool2(x2)
        x3 = self.conv3(x2)
        x3 = self.pool3(x3)

        # Assemble sequence for LSTM
        seq = torch.stack([x1, x2, x3], dim=1)  # (batch, seq_len, features)
        lstm_out, _ = self.lstm(seq)
        out = self.head(lstm_out[:, -1, :])
        return torch.sigmoid(out)

def QCNNHybridFactory() -> QCNNHybrid:
    """Return a ready‑to‑train QCNNHybrid instance."""
    return QCNNHybrid()

__all__ = ["QCNNHybrid", "QCNNHybridFactory"]
