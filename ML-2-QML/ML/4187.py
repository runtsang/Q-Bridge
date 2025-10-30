import torch
import torch.nn as nn
import torch.nn.functional as F
from Autoencoder import AutoencoderNet, AutoencoderConfig
from FCL import FCL

class HybridQLSTMTagger(nn.Module):
    """Hybrid classical LSTM tagger that compresses word embeddings with a classical
    autoencoder, processes them with a standard LSTM, and maps the hidden state to
    tag logits via a bank of classical FCLs (one per tag).  The interface mirrors
    the quantum‑enabled version so that the same class name can be swapped for
    the QML implementation."""
    def __init__(self,
                 vocab_size: int,
                 tagset_size: int,
                 embedding_dim: int = 128,
                 hidden_dim: int = 256,
                 latent_dim: int = 64,
                 n_qubits: int = 0) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.autoencoder = AutoencoderNet(
            config=AutoencoderConfig(
                input_dim=embedding_dim,
                latent_dim=latent_dim,
                hidden_dims=(128, 64),
                dropout=0.1,
            )
        )
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        # One classical FCL per tag; re‑initialise to accept hidden_dim inputs
        self.fcls = nn.ModuleList([FCL() for _ in range(tagset_size)])
        for f in self.fcls:
            f.linear = nn.Linear(hidden_dim, 1)
        self.tagset_size = tagset_size

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        compressed = self.autoencoder.encode(embeds)
        lstm_out, _ = self.lstm(compressed.unsqueeze(0))
        hidden = lstm_out.squeeze(0)
        logits = torch.cat([f.linear(hidden).squeeze(-1) for f in self.fcls], dim=-1)
        return F.log_softmax(logits, dim=-1)

__all__ = ["HybridQLSTMTagger"]
