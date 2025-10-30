"""Classical CNN‑LSTM sequence tagger inspired by Quantum‑NAT and QLSTM.

The architecture first applies a 1‑D convolutional feature extractor to the word embeddings,
then passes the reduced sequence to an LSTM.  The final hidden state is mapped to tag logits.
This design keeps all operations on a classical CPU/GPU and can be trained with standard
optimizers.

Typical usage:

    model = QLSTMGen102(embedding_dim=100, hidden_dim=128, vocab_size=5000,
                        tagset_size=12, n_qubits=0)   # n_qubits ignored in the ML version
    logits = model(sentence_tensor)  # shape (seq_len, tagset_size)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class QLSTMGen102(nn.Module):
    """CNN‑LSTM sequence tagger.

    Parameters
    ----------
    embedding_dim : int
        Size of the word embeddings.
    hidden_dim : int
        Hidden size of both the CNN and the LSTM.
    vocab_size : int
        Size of the vocabulary.
    tagset_size : int
        Number of output tags.
    n_qubits : int, optional
        Unused in the classical version; kept for API compatibility.
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
        # Embedding layer
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # 1‑D CNN feature extractor (mirrors the classical part of Quantum‑NAT)
        self.cnn = nn.Sequential(
            nn.Conv1d(embedding_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # LSTM on the reduced sequence
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, batch_first=True, bidirectional=False
        )

        # Linear projection to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        # Optional dropout mimicking a quantum‑style stochastic gate
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        sentence : torch.Tensor
            Tensor of shape (seq_len, batch) holding word indices.

        Returns
        -------
        torch.Tensor
            Log‑probabilities over tags of shape (seq_len, batch, tagset_size).
        """
        batch_size = sentence.size(1)

        # Embedding
        embeds = self.word_embeddings(sentence)  # (seq_len, batch, embed)

        # Prepare for Conv1d: (batch, embed, seq_len)
        conv_input = embeds.permute(1, 2, 0)

        # CNN feature extraction
        conv_out = self.cnn(conv_input)  # (batch, hidden_dim, seq_len//4)

        # Collapse pooled dimension: (batch, seq_len//4, hidden_dim)
        seq_len_reduced = conv_out.size(2)
        conv_out = conv_out.permute(0, 2, 1).contiguous()

        # LSTM expects (batch, seq_len, hidden)
        lstm_out, _ = self.lstm(conv_out)

        # Map to tag logits
        tag_logits = self.hidden2tag(lstm_out)  # (batch, seq_len_reduced, tagset_size)

        # Pad back to original sequence length if necessary
        if seq_len_reduced!= sentence.size(0):
            pad_len = sentence.size(0) - seq_len_reduced
            padding = torch.zeros(
                batch_size, pad_len, tag_logits.size(-1),
                device=tag_logits.device
            )
            tag_logits = torch.cat([tag_logits, padding], dim=1)

        # Log‑softmax for NLL loss
        return F.log_softmax(tag_logits, dim=-1)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"embedding_dim={self.word_embeddings.embedding_dim}, "
            f"hidden_dim={self.lstm.hidden_size}, "
            f"vocab_size={self.word_embeddings.num_embeddings}, "
            f"tagset_size={self.hidden2tag.out_features})"
        )


__all__ = ["QLSTMGen102"]
