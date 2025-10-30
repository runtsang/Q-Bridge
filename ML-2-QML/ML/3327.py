"""Hybrid classical LSTM with optional convolutional preprocessing.

This module extends the original pure‑PyTorch QLSTM to accept a
classical convolutional pre‑processor.  The `HybridQLSTM` class
acts as a drop‑in replacement for the quantum version and can be
instantiated with `conv_type="classical"` or `"none"`.  It keeps the
same public API used by downstream taggers and is fully compatible
with the original `QLSTM.py` anchor.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class _ConvFilter(nn.Module):
    """Simple MLP‑based filter that mimics a quantum filter."""
    def __init__(self, input_dim: int, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, input_dim)
        out = self.mlp(x)  # (batch, seq_len, 1)
        return out

class HybridQLSTM(nn.Module):
    """Hybrid classical LSTM that optionally prepends a 1‑D convolution."""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_qubits: int = 0,
        conv_type: str = "none",
        conv_kernel: int = 2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.conv_type = conv_type
        self.conv_kernel = conv_kernel

        # Convolution pre‑processor
        if conv_type == "classical":
            self.conv = _ConvFilter(input_dim, kernel_size=conv_kernel)
        else:
            self.conv = None

        # Core LSTM
        if n_qubits > 0:
            # In the classical build we fall back to a pure nn.LSTM;
            # the quantum variant is provided by the qml module.
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        else:
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def _apply_conv(self, seq: torch.Tensor) -> torch.Tensor:
        """Apply the classical 1‑D convolution to each time‑step."""
        if self.conv is None:
            return seq
        conv_out = self.conv(seq)  # (batch, seq_len, 1)
        conv_out = conv_out.expand(-1, -1, self.input_dim)
        return seq + conv_out

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        inputs = self._apply_conv(inputs)
        outputs, (h_n, c_n) = self.lstm(inputs, states)
        return outputs, (h_n, c_n)

class LSTMTagger(nn.Module):
    """Sequence tagging model that uses HybridQLSTM."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        conv_type: str = "none",
        conv_kernel: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = HybridQLSTM(
            embedding_dim,
            hidden_dim,
            n_qubits=n_qubits,
            conv_type=conv_type,
            conv_kernel=conv_kernel,
        )
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=-1)

__all__ = ["HybridQLSTM", "LSTMTagger"]
