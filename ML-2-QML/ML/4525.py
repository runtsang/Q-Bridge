"""Hybrid classical neural architecture combining convolution, LSTM, and kernel-based classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HybridNATModel(nn.Module):
    """Hybrid classical model integrating a convolutional feature extractor, an LSTM encoder, and a kernel-based classifier.

    The architecture is inspired by the QuantumNAT, QLSTM, Quanvolution, and QuantumKernelMethod examples.
    It offers a flexible interface that can be extended with quantum modules while remaining fully classical.
    """

    def __init__(
        self,
        use_quantum_filter: bool = False,
        use_quantum_lstm: bool = False,
        n_qubits: int = 4,
        hidden_dim: int = 64,
        num_classes: int = 10,
        kernel_gamma: float = 1.0,
        n_support: int = 32,
    ) -> None:
        super().__init__()
        # Convolutional feature extractor
        if use_quantum_filter:
            # Placeholder for a quantum‑inspired convolution; here we use a simple 2x2 conv
            self.filter = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        else:
            self.filter = nn.Conv2d(1, 4, kernel_size=2, stride=2)

        # Sequence length after convolution: 14x14 patches -> 196
        self.sequence_len = 196
        self.hidden_dim = hidden_dim

        # LSTM encoder
        if use_quantum_lstm:
            # In classical mode we still use a standard LSTM; quantum LSTM is provided in the QML module.
            self.lstm = nn.LSTM(input_size=4, hidden_size=hidden_dim, batch_first=True)
        else:
            self.lstm = nn.LSTM(input_size=4, hidden_size=hidden_dim, batch_first=True)

        # Kernel‑based classifier
        self.kernel_gamma = kernel_gamma
        # Learnable support vectors
        self.support_vectors = nn.Parameter(torch.randn(n_support, hidden_dim))
        self.classifier_weights = nn.Parameter(torch.randn(n_support, num_classes))

    def _rbf_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute RBF kernel matrix between two batches."""
        diff = x.unsqueeze(1) - y.unsqueeze(0)
        dist_sq = torch.sum(diff * diff, dim=2)
        return torch.exp(-self.kernel_gamma * dist_sq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch, 1, 28, 28)

        Returns:
            Log‑probabilities of shape (batch, num_classes)
        """
        bsz = x.size(0)
        # Convolution
        features = self.filter(x)  # (bsz, 4, 14, 14)
        seq = features.view(bsz, 4, -1).permute(0, 2, 1)  # (bsz, 196, 4)
        # LSTM
        lstm_out, _ = self.lstm(seq)  # (bsz, 196, hidden_dim)
        # Take last hidden state
        last_hidden = lstm_out[:, -1, :]  # (bsz, hidden_dim)
        # Kernel similarity with support vectors
        kernel_mat = self._rbf_kernel(last_hidden, self.support_vectors)  # (bsz, n_support)
        logits = kernel_mat @ self.classifier_weights  # (bsz, num_classes)
        return F.log_softmax(logits, dim=1)

__all__ = ["HybridNATModel"]
