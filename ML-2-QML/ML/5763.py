"""Pure PyTorch implementation of a hybrid LSTM tagger with optional quantum gating.

The model mirrors the interface of the quantum counterpart but uses
pure PyTorch layers.  The ``n_qubits`` flag is accepted for API
compatibility but does not alter the computation.
"""

from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class QLSTMGen(nn.Module):
    """Drop‑in replacement for a classical LSTM tagger.

    Parameters
    ----------
    embedding_dim : int
        Size of word embeddings.
    hidden_dim : int
        Hidden state dimensionality of the LSTM.
    vocab_size : int
        Number of distinct tokens.
    tagset_size : int
        Number of output tags.
    n_qubits : int, default 0
        Unused in the classical version; retained for compatibility.
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
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tagset_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        sentence : torch.Tensor
            LongTensor of shape ``(batch, seq_len)`` containing token indices.

        Returns
        -------
        torch.Tensor
            Log‑probabilities of shape ``(batch, seq_len, tagset_size)``.
        """
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        logits = self.fc(lstm_out)
        return self.log_softmax(logits)


__all__ = ["QLSTMGen"]
