import torch
import torch.nn as nn
import torch.nn.functional as F
from.Autoencoder import Autoencoder

class HybridQLSTM(nn.Module):
    """
    Classical hybrid model: compresses word embeddings via a feed‑forward autoencoder
    before feeding them to a standard LSTM for sequence tagging.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: tuple[int, int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.autoencoder = Autoencoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)                     # (seq_len, batch, embed)
        flat = embeds.view(-1, embeds.size(-1))                     # (seq_len*batch, embed)
        compressed = self.autoencoder.encode(flat)                 # (seq_len*batch, latent)
        compressed = compressed.view(embeds.size(0), embeds.size(1), -1)
        lstm_out, _ = self.lstm(compressed)
        logits = self.hidden2tag(lstm_out)                          # (seq_len, batch, tagset)
        return F.log_softmax(logits, dim=2)

__all__ = ["HybridQLSTM"]
