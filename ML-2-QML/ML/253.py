"""
QCNNEnhanced: Classical attention‑augmented QCNN.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

class QCNNModel(nn.Module):
    """
    Classical QCNN with a multi‑head self‑attention block.
    The architecture mirrors the original seed but adds an
    attention layer after the last convolution to capture
    long‑range dependencies in the feature vector.
    """
    def __init__(self, num_heads: int = 2) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(
            nn.Linear(8, 16),
            nn.Tanh()
        )
        self.conv1 = nn.Sequential(
            nn.Linear(16, 16),
            nn.Tanh()
        )
        self.pool1 = nn.Sequential(
            nn.Linear(16, 12),
            nn.Tanh()
        )
        self.conv2 = nn.Sequential(
            nn.Linear(12, 8),
            nn.Tanh()
        )
        self.pool2 = nn.Sequential(
            nn.Linear(8, 4),
            nn.Tanh()
        )
        self.conv3 = nn.Sequential(
            nn.Linear(4, 4),
            nn.Tanh()
        )
        # Attention block: embed_dim matches last conv output
        self.attn = nn.MultiheadAttention(embed_dim=4, num_heads=num_heads, batch_first=True)
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classical QCNN.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 8).

        Returns
        -------
        torch.Tensor
            Output logits of shape (batch, 1).
        """
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        # Attention expects (batch, seq_len, embed_dim); seq_len=1
        attn_output, _ = self.attn(x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1))
        # Remove seq_len dim
        x = attn_output.squeeze(1)
        logits = self.head(x)
        return torch.sigmoid(logits)

    def fit(self, X: torch.Tensor, y: torch.Tensor,
            epochs: int = 10, lr: float = 1e-3) -> None:
        """
        Simple training loop using Adam and BCE loss.

        Parameters
        ----------
        X : torch.Tensor
            Training data of shape (n_samples, 8).
        y : torch.Tensor
            Binary labels of shape (n_samples, 1).
        epochs : int, optional
            Number of training epochs.
        lr : float, optional
            Learning rate.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCELoss()
        self.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            preds = self.forward(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

def QCNN() -> QCNNModel:
    """
    Factory returning a fresh instance of :class:`QCNNModel`.
    """
    return QCNNModel()

__all__ = ["QCNN", "QCNNModel"]
