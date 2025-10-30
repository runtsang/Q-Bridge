from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HybridConvQLSTMKernel(nn.Module):
    """Hybrid model that chains a classical convolutional filter, a classical LSTM
    encoder, and a radial basis function kernel.  It is designed to be a drop‑in
    replacement for the original ``Conv`` filter while providing higher‑level
    sequence modelling and similarity estimation.

    Parameters
    ----------
    kernel_size : int, default=2
        Size of the 2‑D convolution window.
    lstm_hidden_dim : int, default=32
        Hidden dimension of the LSTM encoder.
    lstm_layers : int, default=1
        Number of stacked LSTM layers.
    gamma : float, default=1.0
        RBF kernel bandwidth.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        lstm_hidden_dim: int = 32,
        lstm_layers: int = 1,
        gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        self.lstm = nn.LSTM(
            input_size=kernel_size * kernel_size,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
        )
        self.kernel = self._RBFKernel(gamma=gamma)

    class _RBFKernel(nn.Module):
        """Simple RBF kernel implemented in PyTorch."""

        def __init__(self, gamma: float = 1.0) -> None:
            super().__init__()
            self.gamma = gamma

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # Broadcast difference and compute squared Euclidean distance
            diff = x.unsqueeze(1) - y.unsqueeze(0)
            sq_norm = torch.sum(diff * diff, dim=-1)
            return torch.exp(-self.gamma * sq_norm)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        data : torch.Tensor
            Input tensor of shape (batch, 1, H, W) where H = W = kernel_size.

        Returns
        -------
        torch.Tensor
            Output logits from the LSTM encoder.
        """
        conv_out = torch.sigmoid(self.conv(data))  # (batch, 1, 1, 1)
        seq = conv_out.view(conv_out.size(0), -1)  # (batch, 1)
        lstm_out, _ = self.lstm(seq.unsqueeze(1))  # (batch, 1, hidden)
        logits = lstm_out.squeeze(1)  # (batch, hidden)
        return logits

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
        """
        Compute the Gram matrix between two sets of vectors using the RBF kernel.

        Parameters
        ----------
        a, b : torch.Tensor
            Tensors of shape (n_samples, features).

        Returns
        -------
        np.ndarray
            Gram matrix of shape (n_samples_a, n_samples_b).
        """
        with torch.no_grad():
            mat = self.kernel(a, b)
        return mat.cpu().numpy()

__all__ = ["HybridConvQLSTMKernel"]
