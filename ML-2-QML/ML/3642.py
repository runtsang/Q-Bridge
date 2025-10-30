"""Hybrid classical kernel and LSTM implementation.

This module provides a unified interface that mimics the quantum modules but
stays entirely within the classical PyTorch ecosystem. It exposes a
`HybridKernelLSTM` class that can compute an RBF kernel and perform
sequence tagging with a standard LSTM.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import Sequence, Tuple

__all__ = [
    "HybridKernelLSTM",
    "Kernel",
    "kernel_matrix",
    "QLSTM",
    "LSTMTagger",
    "KernalAnsatz",
]

# ----------------------------- Classical kernel utilities -----------------------------
class KernalAnsatz(nn.Module):
    """Classical radial basis function kernel component with learnable gamma."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Wrapper around `KernalAnsatz` providing a convenient forward interface."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Ensure 2‑D tensors
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute Gram matrix between two collections of 1‑D tensors."""
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# ----------------------------- Classical LSTM implementations -----------------------------
class QLSTM(nn.Module):
    """Drop‑in classical replacement for the quantum LSTM gates."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        gate_dim = hidden_dim
        self.forget_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.input_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.update_linear = nn.Linear(input_dim + hidden_dim, gate_dim)
        self.output_linear = nn.Linear(input_dim + hidden_dim, gate_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget_linear(combined))
            i = torch.sigmoid(self.input_linear(combined))
            g = torch.tanh(self.update_linear(combined))
            o = torch.sigmoid(self.output_linear(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        stacked = torch.cat(outputs, dim=0)
        return stacked, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        hx = torch.zeros(batch_size, self.hidden_dim, device=device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=device)
        return hx, cx

class LSTMTagger(nn.Module):
    """Sequence tagging model that can switch between a classical LSTM or a
    placeholder for a quantum version (not available in this module)."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if n_qubits > 0:
            # Quantum path is unavailable in the classical module; fall back.
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        # Forward through LSTM
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

# ----------------------------- Unified interface -----------------------------
class HybridKernelLSTM(nn.Module):
    """
    Combines kernel evaluation and sequence tagging into a single
    configurable module.

    Parameters
    ----------
    gamma : float, optional
        RBF kernel width.
    embedding_dim : int, optional
        Size of word embeddings.
    hidden_dim : int, optional
        LSTM hidden state dimension.
    vocab_size : int, optional
        Vocabulary size for embeddings.
    tagset_size : int, optional
        Number of tags for tagging task.
    n_qubits : int, optional
        Number of qubits. If >0, the quantum implementation is used
        (only available in the quantum module). In the classical
        module this argument is ignored.
    """
    def __init__(
        self,
        gamma: float = 1.0,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        vocab_size: int = 5000,
        tagset_size: int = 10,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        # Kernel
        self.kernel = Kernel(gamma)

        # Tagger
        if n_qubits > 0:
            # Quantum implementation not available in this module
            self.tagger = LSTMTagger(embedding_dim, hidden_dim, vocab_size, tagset_size, n_qubits=0)
        else:
            self.tagger = LSTMTagger(embedding_dim, hidden_dim, vocab_size, tagset_size, n_qubits=0)

    def compute_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return scalar kernel value for a pair of samples."""
        return self.kernel(x, y)

    def compute_kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return kernel_matrix(a, b, gamma=self.kernel.ansatz.gamma.item())

    def tag_sequence(self, sentence: torch.Tensor) -> torch.Tensor:
        """Return log‑probabilities over tags for each token."""
        return self.tagger(sentence)
