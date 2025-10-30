import torch
import torch.nn as nn
import torch.nn.functional as F

class UnifiedQLSTMNet(nn.Module):
    """
    Classical LSTM‑based sequence tagger that mirrors the interface of the
    quantum version. All gates are implemented with standard linear layers.
    """
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int,
                 tagset_size: int, n_qubits: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # In the classical variant we always use a standard LSTM
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=False)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sentence: LongTensor of shape (seq_len, batch)
        Returns:
            log‑probabilities over tagset of shape (seq_len, batch, tagset_size)
        """
        embeds = self.embedding(sentence)  # (seq_len, batch, embed_dim)
        lstm_out, _ = self.lstm(embeds)   # (seq_len, batch, hidden_dim)
        tag_logits = self.hidden2tag(lstm_out)  # (seq_len, batch, tagset_size)
        return F.log_softmax(tag_logits, dim=-1)
