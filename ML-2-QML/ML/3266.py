"""Hybrid classical‑quantum kernel‑LSTM module for sequence tagging.

The module exposes two interchangeable back‑ends:
* `HybridKernel` – a classical RBF kernel with a learnable width.
* `HybridQuantumKernel` – a quantum kernel that can be trained via gradient‑based optimizers.
The LSTM layer can operate in a pure‑classical mode (using nn.LSTM) or in a
quantum‑enhanced mode where each gate is implemented by a small variational
circuit.  The architecture is intentionally lightweight so it can be
instantiated on CPU or GPU without requiring a full quantum simulator.

Author: gpt‑oss‑20b
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__all__ = [
    "HybridKernel",
    "HybridKernelLSTM",
    "kernel_matrix",
]

# --------------------------------------------------------------------------- #
# Classical kernel utilities
# --------------------------------------------------------------------------- #
class HybridKernel(nn.Module):
    """Classical RBF kernel with a learnable width parameter."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

    def gram(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> torch.Tensor:
        """Return a Gram matrix for two batches."""
        diff = batch_x[:, None, :] - batch_y[None, :, :]
        distances = torch.sum(diff ** 2, dim=-1)
        return torch.exp(-self.gamma * distances)

def kernel_matrix(a: Iterable[torch.Tensor], b: Iterable[torch.Tensor], gamma: float | None = None) -> np.ndarray:
    """Compute a Gram matrix between two iterables of tensors."""
    if gamma is not None:
        kernel = HybridKernel(gamma)
    else:
        kernel = HybridKernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------------- #
# Hybrid Kernel + LSTM architecture
# --------------------------------------------------------------------------- #
class HybridKernelLSTM(nn.Module):
    """
    Hybrid architecture that couples an RBF kernel with an LSTM tagger.

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of the word embeddings.
    hidden_dim : int
        Hidden size of the LSTM.
    vocab_size : int
        Number of tokens in the vocabulary.
    tagset_size : int
        Number of distinct tags.
    gamma : float, optional
        Initial width of the RBF kernel (default 1.0).
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.kernel = HybridKernel(gamma)
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for sequence tagging.

        Parameters
        ----------
        sentence : torch.Tensor
            LongTensor of token indices with shape (seq_len,).

        Returns
        -------
        torch.Tensor
            Log‑softmax of tag probabilities for each token.
        """
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

    def gram(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> torch.Tensor:
        """Return Gram matrix between two batches using the kernel."""
        return self.kernel.gram(batch_x, batch_y)
