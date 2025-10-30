"""Hybrid image‑sequence classifier: classical quanvolution + classical LSTM.

This module is a pure‑Python (PyTorch) implementation that mirrors the
quantum version in :mod:`Quanvolution__gen211_qml`.  The architectural
choices are motivated by the original quanvolution experiment – a
2×2 convolution that produces a 4‑dimensional embedding per patch – and
the QLSTM design that replaces the linear gates with quantum circuits.
The classical version keeps the same interface but implements all
operations with standard torch layers, making it suitable for
baseline comparisons or environments where quantum hardware is not
available.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["QuanvolutionFilter", "QLSTM", "QuanvolutionQLSTM"]


class QuanvolutionFilter(nn.Module):
    """Classical 2×2 convolution that produces a 4‑dimensional patch embedding."""

    def __init__(self, in_channels: int = 1, out_channels: int = 4,
                 kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride)
        # Number of patches in a 28×28 image with 2×2 stride 2
        self.num_patches = 14 * 14

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a tensor of shape (B, seq_len, feat_dim)."""
        # (B, C, H, W) -> (B, out_channels, H', W')
        feat = self.conv(x)
        # (B, 4, 14, 14) -> (B, 14, 14, 4)
        feat = feat.permute(0, 2, 3, 1)
        # (B, 14, 14, 4) -> (B, 196, 4)
        return feat.reshape(x.size(0), self.num_patches, -1)


class QLSTM(nn.Module):
    """Classical LSTM cell with the same interface as the quantum version."""

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0) -> None:
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
        # inputs: (seq_len, batch, input_dim)
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
        return torch.cat(outputs, dim=0), (hx, cx)

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


class QuanvolutionQLSTM(nn.Module):
    """Hybrid classifier: quanvolution filter → LSTM → linear head.

    Parameters
    ----------
    num_classes : int
        Number of target classes.
    hidden_dim : int, optional
        Hidden dimension of the LSTM. Defaults to 128.
    """

    def __init__(self, num_classes: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.filter = QuanvolutionFilter()
        self.lstm = QLSTM(input_dim=4, hidden_dim=hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 28, 28)
        seq = self.filter(x)  # (B, seq_len, 4)
        # LSTM expects (seq_len, batch, input_dim)
        seq = seq.permute(1, 0, 2)
        lstm_out, _ = self.lstm(seq)
        # Use the last hidden state as representation
        final_hidden = lstm_out[-1]
        logits = self.classifier(final_hidden)
        return F.log_softmax(logits, dim=-1)
