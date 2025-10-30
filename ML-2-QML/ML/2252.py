"""Hybrid vision‑to‑sequence model that fuses a classical quanvolution filter with a classical LSTM.

The architecture is useful for tasks such as image captioning or video‑frame prediction.
It keeps the classical convolutional front‑end for efficient feature extraction and
uses a classical LSTM cell to propagate information across the sequence of patches.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionFilter(nn.Module):
    """Classical 2×2 patch extraction using a small convolution."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4, patch_size: int = 2) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a flattened feature vector of shape (batch, out_channels * 14 * 14)."""
        features = self.conv(x)  # (B, out_channels, 14, 14)
        return features.view(x.size(0), -1)

class ClassicalQLSTM(nn.Module):
    """Classical LSTM cell that mimics the quantum‑enhanced gates with linear layers."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

    def forward(self, inputs: torch.Tensor, states: tuple | None = None) -> tuple:
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

    def _init_states(self, inputs: torch.Tensor, states: tuple | None = None) -> tuple:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

class QuanvolutionQLSTMHybrid(nn.Module):
    """Hybrid vision‑to‑sequence model combining a classical quanvolution filter and a classical LSTM."""
    def __init__(self, n_qubits: int = 4, hidden_dim: int = 128, num_classes: int = 10) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.lstm = ClassicalQLSTM(input_dim=4, hidden_dim=hidden_dim, n_qubits=n_qubits)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 28, 28)
        batch_size = x.size(0)
        patches = self.qfilter(x)  # (B, 4*14*14)
        seq_len = 14 * 14
        patch_dim = 4
        patches_seq = patches.view(batch_size, seq_len, patch_dim).transpose(0, 1)  # (seq_len, B, patch_dim)
        lstm_out, _ = self.lstm(patches_seq)
        final_hidden = lstm_out[-1]  # (B, hidden_dim)
        logits = self.classifier(final_hidden)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionFilter", "ClassicalQLSTM", "QuanvolutionQLSTMHybrid"]
