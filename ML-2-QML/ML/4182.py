"""Hybrid classical LSTM with a regression head.

This module provides the class :class:`HybridQLSTM` that behaves like the
original QLSTM but keeps all gates classical.  An additional linear head
produces per‑time‑step regression outputs.  The API is identical to the
quantum version, so the same code can be used with either backend.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class HybridQLSTM(nn.Module):
    """Drop‑in classical LSTM with an optional regression head.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    hidden_dim : int
        Size of the hidden state.
    n_qubits : int, default 0
        If >0 the module pretends to use quantum gates; the implementation
        remains classical to keep the API compatible with the quantum
        counterpart.
    output_dim : int, default 1
        Dimensionality of the regression output per time step.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0, output_dim: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.output_dim = output_dim

        # Classical linear gates
        self.forget_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Regression head applied to the hidden state
        self.regressor = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        regress = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)

            outputs.append(hx.unsqueeze(0))
            regress.append(self.regressor(hx).unsqueeze(0))

        stacked = torch.cat(outputs, dim=0)
        regress_out = torch.cat(regress, dim=0)
        return stacked, regress_out, (hx, cx)

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


class RegressionDataset(Dataset):
    """Dataset that generates synthetic superposition‑like data.

    The labels mimic the quantum example: y = sin(∑x) + 0.1 cos(2∑x).
    """
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = self._generate(samples, num_features)

    @staticmethod
    def _generate(samples: int, num_features: int) -> tuple[np.ndarray, np.ndarray]:
        x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
        angles = x.sum(axis=1)
        y = np.sin(angles) + 0.1 * np.cos(2 * angles)
        return x, y.astype(np.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


__all__ = ["HybridQLSTM", "RegressionDataset"]
