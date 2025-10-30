"""Hybrid classical LSTM with optional quantum gating and quanvolution feature extraction.

This file implements a drop‑in replacement for the original QLSTM
that supports three operating modes:

* *tagger* – sequence tagging with a classical LSTM.  When ``use_kernel``
  is True a simple RBF kernel is applied to the LSTM output before the
  final linear layer, providing a learnable non‑linear transformation.

* *classifier* – image classification using a Quanvolution filter
  followed by a linear head.  The filter can be switched to a plain
  Conv2d by setting ``use_quanvolution`` to False.

* *hybrid* – a combination of the two, useful for multimodal problems.

The public API matches the original QLSTM implementation so that
existing training scripts can be swapped in without modification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --------------------------------------------------------------------------- #
#   Classical kernel utilities (from QuantumKernelMethod.py)
# --------------------------------------------------------------------------- #
class KernalAnsatz(nn.Module):
    """Placeholder maintaining compatibility with the quantum interface."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class Kernel(nn.Module):
    """RBF kernel module that wraps :class:`KernalAnsatz`."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()


def kernel_matrix(a: list[torch.Tensor], b: list[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Convenience helper that evaluates a Gram matrix."""
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


# --------------------------------------------------------------------------- #
#   Classical quanvolution filter (from Quanvolution.py)
# --------------------------------------------------------------------------- #
class QuanvolutionFilter(nn.Module):
    """2‑D convolution that mimics the behaviour of the quantum filter."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.conv(x)
        return features.view(x.size(0), -1)  # flatten per sample


# --------------------------------------------------------------------------- #
#   Core hybrid model
# --------------------------------------------------------------------------- #
class HybridQLSTM(nn.Module):
    """
    A hybrid classical/quantum LSTM that can operate either as a
    sequence tagger or as an image classifier.  The design intentionally
    mirrors the public API of the original QLSTM implementation so
    that it can be swapped in with minimal code changes.
    """

    def __init__(
        self,
        mode: str,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        vocab_size: int = 5000,
        tagset_size: int = 10,
        n_qubits: int = 0,
        use_kernel: bool = False,
        use_quanvolution: bool = False,
        gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.use_kernel = use_kernel
        self.use_quanvolution = use_quanvolution
        self.gamma = gamma

        # ------------------------------------------------------------------ #
        #   Tagging branch
        # ------------------------------------------------------------------ #
        if mode in {"tagger", "hybrid"}:
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

            # Classical LSTM is used here; n_qubits is kept for API compatibility
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)

            self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

            if use_kernel:
                # Learnable reference vector for the RBF kernel
                self.kernel_vector = nn.Parameter(torch.randn(hidden_dim))
        else:
            self.word_embeddings = None

        # ------------------------------------------------------------------ #
        #   Classifier branch
        # ------------------------------------------------------------------ #
        if mode in {"classifier", "hybrid"}:
            if use_quanvolution:
                self.qfilter = QuanvolutionFilter()
            else:
                # Classical fallback
                self.qfilter = nn.Conv2d(1, 4, kernel_size=2, stride=2)

            self.linear = nn.Linear(4 * 14 * 14, 10)
        else:
            self.qfilter = None

    # ---------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.mode == "tagger":
            return self._forward_tagger(x)
        if self.mode == "classifier":
            return self._forward_classifier(x)
        if self.mode == "hybrid":
            tag_logits = self._forward_tagger(x)
            return self._forward_classifier(tag_logits)
        raise ValueError(f"Unsupported mode: {self.mode}")

    # ---------------------------------------------------------------------- #
    def _forward_tagger(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        if self.use_kernel:
            # Simple RBF gating signal applied element‑wise
            gate = torch.exp(-self.gamma * torch.sum((lstm_out - self.kernel_vector) ** 2, dim=-1, keepdim=True))
            lstm_out = lstm_out * gate
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=-1)

    # ---------------------------------------------------------------------- #
    def _forward_classifier(self, image: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(image)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["HybridQLSTM"]
