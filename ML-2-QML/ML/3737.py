"""Hybrid kernel and LSTM module for classical usage."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["HybridKernelLSTM"]


class ClassicalRBFKernel(nn.Module):
    """Efficient, differentiable RBF kernel for GPU execution."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return exp(-gamma * ||x-y||^2) for each pair."""
        # Broadcast to (len(x), len(y), dim)
        diff = x.unsqueeze(1) - y.unsqueeze(0)
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1))


class KernelWrapper(nn.Module):
    """Build a Gram matrix from a kernel module."""
    def __init__(self, kernel: nn.Module) -> None:
        super().__init__()
        self.kernel = kernel

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> np.ndarray:
        """Return Gram matrix as numpy array."""
        gram = self.kernel(x, y).cpu().numpy()
        return gram


class HybridKernelLSTM(nn.Module):
    """Hybrid module that offers a classical RBF kernel and a classical LSTM tagger."""
    def __init__(
        self,
        *,
        kernel_gamma: float = 1.0,
        embedding_dim: int = 50,
        hidden_dim: int = 128,
        vocab_size: int = 10000,
        tagset_size: int = 10,
    ) -> None:
        super().__init__()
        self.kernel = KernelWrapper(ClassicalRBFKernel(gamma=kernel_gamma))
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
        """Convenience wrapper for kernel Gram matrix."""
        return self.kernel(a, b)

    def tag_sequence(self, sentence: torch.LongTensor) -> torch.Tensor:
        """
        Forward pass for sequence tagging.

        Parameters
        ----------
        sentence : LongTensor, shape (seq_len,)
            Token indices of the input sentence.

        Returns
        -------
        log_probs : Tensor, shape (seq_len, tagset_size)
            Log‑probabilities over tags for each token.
        """
        embeds = self.embedding(sentence).unsqueeze(0)  # batch size 1
        lstm_out, _ = self.lstm(embeds)  # shape (1, seq_len, hidden_dim)
        logits = self.hidden2tag(lstm_out.squeeze(0))  # shape (seq_len, tagset_size)
        return F.log_softmax(logits, dim=1)
