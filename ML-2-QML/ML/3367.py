"""Hybrid classical LSTM + regressor for sequence tagging and regression.

This module defines a drop‑in replacement that mirrors the interface of
the original QLSTM but extends it with a regression head.  The core of the
model is a standard PyTorch LSTM used as a feature extractor, followed by
a small fully‑connected network that predicts a scalar value from the final
hidden state.  The design is deliberately simple to keep the forward pass
efficient and to serve as a baseline for quantum‑enhanced variants.

The class can be used in two modes:
  * ``n_qubits=0`` – pure classical LSTM + linear regressor.
  * ``n_qubits>0`` – delegates to the quantum implementation in
    ``qml_code`` (not imported here to avoid heavy dependencies).

The API matches that of the original QLSTM module so that existing training
pipelines continue to work unchanged.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridQLSTMRegressor(nn.Module):
    """Classical hybrid LSTM + regressor.

    Parameters
    ----------
    embedding_dim : int
        Dimension of word embeddings.
    hidden_dim : int
        Size of LSTM hidden states.
    vocab_size : int
        Number of distinct tokens in the vocabulary.
    tagset_size : int
        Number of target tags for the tagging head.
    n_qubits : int, default 0
        If ``>0`` the class will raise a RuntimeError indicating that a
        quantum variant is required.  This keeps the API compatible with the
        quantum module while avoiding unnecessary imports.
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
        if n_qubits > 0:
            raise RuntimeError(
                "HybridQLSTMRegressor is the classical variant; "
                "use the quantum module for n_qubits > 0."
            )
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        # Regression head: uses the final hidden state
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, sentence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        sentence : torch.Tensor
            LongTensor of shape (seq_len, batch) containing token indices.

        Returns
        -------
        tag_logits : torch.Tensor
            Log‑softmax over tags, shape (seq_len, batch, tagset_size).
        regression : torch.Tensor
            Scalar predictions, shape (batch, 1).
        """
        embeds = self.word_embeddings(sentence)
        lstm_out, (hn, _) = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        tag_logits = F.log_softmax(tag_logits, dim=-1)
        regression = self.regressor(hn.squeeze(0))
        return tag_logits, regression


__all__ = ["HybridQLSTMRegressor"]
