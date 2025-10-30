import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class AutoencoderNet(nn.Module):
    """Lightweight fully‑connected autoencoder."""
    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dims: Tuple[int, int] = (128, 64), dropout: float = 0.1) -> None:
        super().__init__()
        encoder = []
        in_dim = input_dim
        for h in hidden_dims:
            encoder.append(nn.Linear(in_dim, h))
            encoder.append(nn.ReLU())
            if dropout > 0.0:
                encoder.append(nn.Dropout(dropout))
            in_dim = h
        encoder.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder)

        decoder = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder.append(nn.Linear(in_dim, h))
            decoder.append(nn.ReLU())
            if dropout > 0.0:
                decoder.append(nn.Dropout(dropout))
            in_dim = h
        decoder.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

class UnifiedQLSTMNet(nn.Module):
    """
    Classical sequence‑tagger that optionally stitches a fully‑connected
    autoencoder before a conventional LSTM and a linear decoder.
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        use_autoencoder: bool = False,
        autoencoder_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.use_autoencoder = use_autoencoder
        if use_autoencoder:
            cfg = autoencoder_cfg or {"input_dim": embedding_dim}
            self.autoencoder = AutoencoderNet(**cfg)
            feat_dim = cfg.get("latent_dim", 32)
        else:
            self.autoencoder = None
            feat_dim = embedding_dim
        self.lstm = nn.LSTM(feat_dim, hidden_dim, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        sentence : torch.LongTensor, shape (batch, seq_len)
            Word indices.

        Returns
        -------
        log_probs : torch.Tensor, shape (batch, seq_len, tagset_size)
            Log‑probabilities over tags.
        """
        embeds = self.word_embeddings(sentence)
        if self.use_autoencoder:
            embeds = self.autoencoder(embeds)
        lstm_out, _ = self.lstm(embeds)
        logits = self.hidden2tag(lstm_out)
        return F.log_softmax(logits, dim=-1)

__all__ = ["UnifiedQLSTMNet", "AutoencoderNet"]
