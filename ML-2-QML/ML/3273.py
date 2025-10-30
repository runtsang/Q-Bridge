"""Classical hybrid kernel and LSTM for sequence tagging.

This module implements:
- A differentiable RBF kernel with optional trainable gamma.
- A classical LSTM tagger.
- Utility to compute the Gram matrix.
- Backward‑compatibility aliases for the original API.

The design follows the original QuantumKernelMethod and QLSTM seeds but removes
all quantum dependencies, enabling pure‑CPU training.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class RBFKernel(nn.Module):
    """Classical RBF kernel with optional trainable gamma."""
    def __init__(self, gamma: float | None = None, trainable: bool = False):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma if gamma is not None else 1.0))
        if not trainable:
            self.gamma.requires_grad = False

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor],
                  gamma: float = 1.0, trainable: bool = False) -> np.ndarray:
    """Return Gram matrix for pairs of tensors using an RBF kernel."""
    kernel = RBFKernel(gamma, trainable=trainable)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

class ClassicalLSTMTagger(nn.Module):
    """Sequence tagging model using a classical LSTM."""
    def __init__(self, embedding_dim: int, hidden_dim: int,
                 vocab_size: int, tagset_size: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence).unsqueeze(0)  # batch size = 1
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out.squeeze(0))
        return F.log_softmax(tag_logits, dim=1)

class HybridKernelLSTM(nn.Module):
    """Wrapper that exposes both kernel and LSTM components."""
    def __init__(self, embedding_dim: int, hidden_dim: int,
                 vocab_size: int, tagset_size: int,
                 gamma: float = 1.0, kernel_trainable: bool = False):
        super().__init__()
        self.kernel = RBFKernel(gamma, trainable=kernel_trainable)
        self.tagger = ClassicalLSTMTagger(embedding_dim, hidden_dim,
                                          vocab_size, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        return self.tagger(sentence)

    def gram_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return kernel_matrix(a, b, gamma=self.kernel.gamma.item(),
                             trainable=self.kernel.gamma.requires_grad)

# Backward‑compatibility aliases
KernalAnsatz = RBFKernel
Kernel = HybridKernelLSTM
QLSTM = ClassicalLSTMTagger
LSTMTagger = HybridKernelLSTM

__all__ = ["RBFKernel", "kernel_matrix", "ClassicalLSTMTagger",
           "HybridKernelLSTM", "KernalAnsatz", "Kernel", "QLSTM", "LSTMTagger"]
