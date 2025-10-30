"""Classical hybrid model combining convolution, LSTM, attention, and regression head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionHybrid(nn.Module):
    """
    Classical counterpart of the quantum hybrid architecture.
    Uses a 2‑D convolution as a proxy for the quantum filter,
    a standard LSTM for sequence modelling,
    a linear attention mechanism,
    and a linear regressor inspired by EstimatorQNN.
    """

    def __init__(self, hidden_dim: int = 32, attention_dim: int = 32) -> None:
        super().__init__()
        # Convolution mimicking a 2x2 patch filter
        self.qfilter = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        # LSTM over the sequence of 196 patches
        self.lstm = nn.LSTM(input_size=4, hidden_size=hidden_dim, batch_first=True)
        # Simple linear attention that learns a weight per hidden unit
        self.attn = nn.Linear(hidden_dim, attention_dim)
        # Linear regression head
        self.estimator = nn.Linear(attention_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Batch of grayscale images of shape (B, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Regression predictions of shape (B, 1).
        """
        # 1️⃣ Convolution → feature map of shape (B, 4, 14, 14)
        features = self.qfilter(x)
        # 2️⃣ Reshape to a sequence of 196 patches, each 4‑dimensional
        seq = features.view(x.size(0), 4, 14 * 14).permute(0, 2, 1)  # (B, 196, 4)
        # 3️⃣ LSTM over the sequence
        lstm_out, _ = self.lstm(seq)  # (B, 196, hidden_dim)
        # 4️⃣ Aggregate (mean) over time
        lstm_mean = lstm_out.mean(dim=1)  # (B, hidden_dim)
        # 5️⃣ Attention weighting
        attn_weights = F.softmax(self.attn(lstm_mean), dim=-1)  # (B, attention_dim)
        # 6️⃣ Weighted sum (in this simple case it's just the attention vector)
        attn_out = attn_weights  # (B, attention_dim)
        # 7️⃣ Regression head
        out = self.estimator(attn_out)  # (B, 1)
        return out


__all__ = ["QuanvolutionHybrid"]
