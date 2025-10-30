from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridSequenceClassifier(nn.Module):
    """
    Classical LSTM‑based binary classifier.
    Embeddings → LSTM → Linear head → Sigmoid.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids : torch.LongTensor, shape (batch, seq_len)
            Token indices.

        Returns
        -------
        torch.Tensor, shape (batch, num_classes)
            Probabilities for each class.
        """
        embedded = self.embedding(input_ids)          # (batch, seq_len, embed_dim)
        lstm_out, _ = self.lstm(embedded)            # (batch, seq_len, hidden_dim)
        last_hidden = lstm_out[:, -1, :]              # (batch, hidden_dim)
        hidden = self.dropout(last_hidden)
        logits = self.fc(hidden)                     # (batch, num_classes)
        probs = self.sigmoid(logits)                 # (batch, num_classes)
        return probs

__all__ = ["HybridSequenceClassifier"]
