import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class HybridLSTMTagger(nn.Module):
    """
    Classical LSTM‑based sequence tagger.
    This class mirrors the interface of the original LSTMTagger but
    is fully classical. It accepts a ``n_qubits`` argument for
    API compatibility but ignores it when building the model.
    """
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 tagset_size: int,
                 n_qubits: int = 0) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # Use a standard PyTorch LSTM
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a batch of sentences.
        ``sentence`` is expected to be of shape (seq_len, batch).
        """
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=-1)

__all__ = ["HybridLSTMTagger"]
