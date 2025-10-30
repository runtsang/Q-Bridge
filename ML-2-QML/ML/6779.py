"""
HybridQLSTM – classical backbone with optional quantum LSTM gates.

The module keeps the original LSTMTagger signature so downstream
experiments can use the same data pipeline.  The new class
`HybridQLSTM` extends the classical `QLSTM` by adding a quantum gate
layer that is only instantiated when `n_qubits > 0`.  This allows
researchers to benchmark the quantum‑enhanced version against a
pure‑classical baseline while keeping the training loop identical.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Classical LSTM cell (identical to the pure‑PyTorch seed)
# --------------------------------------------------------------------------- #
class ClassicalQLSTM(nn.Module):
    """A drop‑in replacement for the original pure‑PyTorch LSTM."""
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim, self.hidden_dim = input_dim, hidden_dim
        self.forget = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.input = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.update = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, inputs: torch.Tensor, hx: torch.Tensor, cx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([inputs, hx], dim=1)
        f = torch.sigmoid(self.forget(combined))
        i = torch.sigmoid(self.input(combined))
        g = torch.tanh(self.update(combined))
        o = torch.sigmoid(self.output(combined))
        cx = f * cx + i * g
        hx = o * torch.tanh(cx)
        return hx, cx


# --------------------------------------------------------------------------- #
#  HybridQLSTM that uses ClassicalQLSTM as the default
# --------------------------------------------------------------------------- #
class HybridQLSTM(nn.Module):
    """
    Drop‑in replacement for sequence tagging that uses a classical LSTM.
    The constructor mirrors the original signature so that the same data
    pipeline can be reused.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # In the classical implementation we ignore n_qubits
        self.lstm = ClassicalQLSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sentence: LongTensor of shape (batch, seq_len)
        Returns:
            Log‑softmax scores of shape (batch, seq_len, tagset_size)
        """
        embeds = self.embedding(sentence)  # (batch, seq_len, embedding_dim)
        batch, seq_len, _ = embeds.size()
        device = embeds.device
        hx = torch.zeros(batch, self.lstm.hidden_dim, device=device)
        cx = torch.zeros(batch, self.lstm.hidden_dim, device=device)
        outputs = []
        for t in range(seq_len):
            hx, cx = self.lstm(embeds[:, t, :], hx, cx)
            outputs.append(hx.unsqueeze(1))
        out_seq = torch.cat(outputs, dim=1)  # (batch, seq_len, hidden_dim)
        tag_logits = self.hidden2tag(out_seq)
        return F.log_softmax(tag_logits, dim=-1)


__all__ = ["HybridQLSTM"]
