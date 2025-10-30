import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalQLSTMTagger(nn.Module):
    """
    Purely classical LSTM tagger that mirrors the interface of the quantum version.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sentence: LongTensor of shape (seq_len,) with token indices.
        Returns:
            Log‑softmax scores over tags for each token.
        """
        emb = self.embedding(sentence).unsqueeze(1)  # (seq_len, 1, emb_dim)
        out, _ = self.lstm(emb)                      # (seq_len, 1, hidden_dim)
        logits = self.classifier(out.squeeze(1))     # (seq_len, tagset_size)
        return F.log_softmax(logits, dim=-1)

__all__ = ["ClassicalQLSTMTagger"]
