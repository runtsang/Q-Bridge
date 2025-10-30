"""Hybrid kernel‑LSTM module for classical experiments."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

__all__ = ["HybridKernelLSTM"]


class _RBFKernel(nn.Module):
    """Fast analytic RBF kernel."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return exp(-gamma * ||x - y||^2)."""
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class HybridKernelLSTM(nn.Module):
    """
    A hybrid architecture that combines a classical RBF kernel with a shared LSTM encoder.
    The class is intentionally lightweight so that it can be used as a drop‑in replacement
    for the original ``QuantumKernelMethod`` and ``QLSTM`` implementations.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input feature vectors.
    hidden_dim : int
        Hidden size of the LSTM encoder.
    rbf_gamma : float, default 1.0
        Width parameter for the RBF kernel.
    support_vectors : torch.Tensor, optional
        Tensor of shape [num_support, input_dim] that will be used
        to form the kernel Gram matrix.  If ``None`` no kernel
        computation is performed.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        rbf_gamma: float = 1.0,
        support_vectors: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.rbf_kernel = _RBFKernel(gamma=rbf_gamma)
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.support_vectors = support_vectors

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence of shape [seq_len, batch, input_dim].

        Returns
        -------
        lstm_out : torch.Tensor
            LSTM hidden states of shape [seq_len, batch, hidden_dim].
        kernel_matrix : torch.Tensor
            Gram matrix between the flattened input and the support vectors
            of shape [seq_len * batch, num_support].  If no support vectors
            were supplied, an empty tensor is returned.
        """
        seq_len, batch, _ = x.shape
        x_flat = x.reshape(seq_len * batch, -1)

        if self.support_vectors is not None:
            sv = self.support_vectors.to(x.device)
            # Classical RBF kernel
            diff = x_flat.unsqueeze(1) - sv.unsqueeze(0)  # [N, M, D]
            kernel_matrix = torch.exp(
                -self.rbf_kernel.gamma * torch.sum(diff * diff, dim=-1)
            )
        else:
            kernel_matrix = torch.empty((0, 0), device=x.device)

        lstm_out, _ = self.lstm(x)
        return lstm_out, kernel_matrix
