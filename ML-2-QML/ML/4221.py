"""Hybrid classical classifier that augments a feed‑forward network with
a radial‑basis kernel feature map and optional LSTM for sequence data.
The design mirrors the quantum helper interface but remains pure
PyTorch."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QuantumHybridClassifier(nn.Module):
    """
    Classical counterpart of the quantum hybrid architecture.
    Parameters
    ----------
    input_dim : int
        Dimensionality of the raw feature vector.
    hidden_dim : int
        Width of the hidden layer in the feed‑forward network.
    num_support : int, default 16
        Number of support vectors used for the kernel expansion.
    gamma : float, default 1.0
        RBF kernel bandwidth.
    use_lstm : bool, default False
        When True a 1‑layer LSTM processes sequential inputs.
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_support: int = 16,
                 gamma: float = 1.0,
                 use_lstm: bool = False):
        super().__init__()
        self.use_lstm = use_lstm
        if use_lstm:
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            feature_dim = hidden_dim
        else:
            feature_dim = input_dim

        self.kernel = lambda x, s: torch.exp(-gamma * torch.sum((x - s) ** 2, dim=1, keepdim=True))
        # initialise support vectors as learnable parameters
        self.support = nn.Parameter(torch.randn(num_support, feature_dim))
        self.kernel_dim = num_support

        # feed‑forward head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim + num_support, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, seq_len, feat) if use_lstm else (batch, feat).
        Returns
        -------
        logits : torch.Tensor
            Shape (batch, 2)
        """
        if self.use_lstm:
            # x shape: (batch, seq_len, feat)
            _, (h_n, _) = self.lstm(x)
            feat = h_n.squeeze(0)  # (batch, hidden_dim)
        else:
            feat = x  # (batch, feat)

        # kernel expansion
        batch_size = feat.shape[0]
        # compute kernel between each sample and each support vector
        # feat: (batch, dim), support: (num_support, dim)
        diff = feat.unsqueeze(1) - self.support.unsqueeze(0)  # (batch, num_support, dim)
        k = torch.exp(-torch.sum(diff ** 2, dim=2))  # (batch, num_support)
        # concatenate
        combined = torch.cat([feat, k], dim=1)  # (batch, dim+num_support)
        logits = self.classifier(combined)
        return logits

__all__ = ["QuantumHybridClassifier"]
