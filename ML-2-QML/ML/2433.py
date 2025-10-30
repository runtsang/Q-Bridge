"""Hybrid classical model combining CNN feature extraction and optional quantum refinement."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["HybridNATQLSTM"]

class HybridNATQLSTM(nn.Module):
    """Hybrid CNN + optional quantum refinement + LSTM for sequence modeling."""
    def __init__(self, hidden_dim: int = 128, refine: bool = True):
        super().__init__()
        self.backbone = self._CNNBackbone()
        self.refine = refine
        if refine:
            self.refine_layer = self._QuantumRefine()
        self.lstm = nn.LSTM(input_size=4, hidden_size=hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, 4)

    class _CNNBackbone(nn.Module):
        """Convolutional feature extractor producing a 4‑dimensional vector."""
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.fc = nn.Sequential(
                nn.Linear(16 * 7 * 7, 64),
                nn.ReLU(),
                nn.Linear(64, 4),
            )
            self.norm = nn.BatchNorm1d(4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            bsz = x.size(0)
            feat = self.features(x)
            flat = feat.view(bsz, -1)
            out = self.fc(flat)
            return self.norm(out)

    class _QuantumRefine(nn.Module):
        """Light‑weight variational refinement of the 4‑dim vector."""
        def __init__(self, n_wires: int = 4) -> None:
            super().__init__()
            # Simple parameterised linear layers emulating RX, RY, RZ
            self.linear_rx = nn.Linear(n_wires, n_wires)
            self.linear_ry = nn.Linear(n_wires, n_wires)
            self.linear_rz = nn.Linear(n_wires, n_wires)
            self.norm = nn.BatchNorm1d(n_wires)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape: (batch, 4)
            out = self.linear_rx(x)
            out = self.linear_ry(out)
            out = self.linear_rz(out)
            return self.norm(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, 1, 28, 28) representing a
            sequence of grayscale images.

        Returns
        -------
        torch.Tensor
            Logits of shape (batch, seq_len, 4) after the LSTM and final
            linear layer.
        """
        batch, seq_len, *rest = x.shape
        x = x.view(batch * seq_len, *rest)  # (batch*seq_len, 1, 28, 28)
        feats = self.backbone(x)  # (batch*seq_len, 4)
        if self.refine:
            feats = self.refine_layer(feats)  # (batch*seq_len, 4)
        feats = feats.view(batch, seq_len, -1)  # (batch, seq_len, 4)
        lstm_out, _ = self.lstm(feats)  # (batch, seq_len, hidden_dim)
        out = self.output_layer(lstm_out)  # (batch, seq_len, 4)
        return out
