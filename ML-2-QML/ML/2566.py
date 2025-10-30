"""Unified hybrid estimator that merges classical and quantum LSTM modules.

The module defines a single ``UnifiedEstimatorQLSTM`` class that
* uses a classical fully‑connected regressor to produce a scalar output
* optionally wraps a quantum‑enhanced LSTM (QLSTM) for sequence tagging
* exposes a ``forward`` method that accepts either a sequence or a
  2‑D tensor and dispatches to the appropriate sub‑module.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class _FCRegressor(nn.Module):
    """A small fully‑connected network that mimics the EstimatorQNN
    implementation but with a tunable hidden size and a residual
    connection for improved gradient flow."""
    def __init__(self, input_dim: int, hidden_size: int = 8, bias: bool = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size, bias=bias),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size, bias=bias),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=bias),
        )
        self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.residual(x)


class UnifiedEstimatorQLSTM(nn.Module):
    """Hybrid estimator that can run a classical regressor or a quantum‑enhanced LSTM tagger.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input for the regression head.
    hidden_dim : int
        Hidden size used in the LSTM tagger.
    vocab_size : int
        Vocabulary size for the embedding layer in the LSTM tagger.
    tagset_size : int
        Number of output tags for sequence tagging.
    n_qubits : int, optional
        If >0, the LSTM tagger will use the quantum LSTM implementation.
        Otherwise, a classical nn.LSTM is used.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        # Classical regression head
        self.regressor = _FCRegressor(input_dim)

        # Sequence tagging head
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, input_dim)
        if n_qubits > 0:
            from.qlstm import QLSTM  # local import to avoid heavy deps
            self.lstm = QLSTM(input_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(
        self,
        data: torch.Tensor,
        *,  # keyword‑only to avoid ambiguity
        mode: str = "regress",
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        data : torch.Tensor
            If ``mode == "regress"``: a 2‑D tensor of shape (batch, features).
            If ``mode == "tagger"``: a 1‑D tensor of word indices (batch, seq_len).
        mode : str
            Select which head to run: ``"regress"`` or ``"tagger"``.
        """
        if mode == "regress":
            return self.regressor(data)
        if mode == "tagger":
            # Prepare embeddings
            embeds = self.word_embeddings(data)
            # LSTM expects (batch, seq_len, features) if batch_first=True
            lstm_out, _ = self.lstm(embeds)
            logits = self.hidden2tag(lstm_out)
            return F.log_softmax(logits, dim=-1)
        raise ValueError(f"Unknown mode {mode!r}")

__all__ = ["UnifiedEstimatorQLSTM"]
