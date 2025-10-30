"""Pure PyTorch implementation with dropout and residual connections."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QLSTM(nn.Module):
    """Drop‑in replacement using a classical LSTM with dropout and residual connections."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=False)
        self.dropout = nn.Dropout(dropout)
        # Residual projection if hidden_dim!= input_dim
        if hidden_dim!= input_dim:
            self.residual_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.residual_proj = None

    def forward(self, inputs):
        """
        inputs: Tensor of shape (seq_len, batch, input_dim)
        Returns: Tensor of shape (seq_len, batch, hidden_dim)
        """
        lstm_out, _ = self.lstm(inputs)
        lstm_out = self.dropout(lstm_out)
        # Residual connection
        if self.residual_proj is not None:
            residual = self.residual_proj(inputs)
        else:
            residual = inputs
        return lstm_out + residual


class LSTMTagger(nn.Module):
    """Sequence tagging model that uses either :class:`QLSTM` or ``nn.LSTM``."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        dropout: float = 0.1,
        use_qlstm: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if use_qlstm:
            self.lstm = QLSTM(embedding_dim, hidden_dim, dropout)
        else:
            # Classic LSTM with dropout
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=False)
            self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        """
        sentence: Tensor of shape (seq_len, batch)
        Returns: Tensor of shape (seq_len, batch, tagset_size)
        """
        embeds = self.word_embeddings(sentence)
        if isinstance(self.lstm, QLSTM):
            lstm_out = self.lstm(embeds)
        else:
            lstm_out, _ = self.lstm(embeds)
            lstm_out = self.dropout(lstm_out)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=-1)


__all__ = ["QLSTM", "LSTMTagger"]
