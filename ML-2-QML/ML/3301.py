import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridAttentionAutoencoder(nn.Module):
    """Classical hybrid model: self‑attention layer followed by a dense autoencoder."""
    def __init__(self,
                 embed_dim: int = 4,
                 latent_dim: int = 32,
                 hidden_dims: tuple = (128, 64),
                 dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        # Self‑attention parameters as learnable weights
        self.rotation_params = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.entangle_params = nn.Parameter(torch.randn(embed_dim, embed_dim))
        # Autoencoder
        encoder_layers = []
        in_dim = embed_dim
        for h in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h))
            encoder_layers.append(nn.ReLU())
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
            in_dim = h
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h))
            decoder_layers.append(nn.ReLU())
            if dropout > 0:
                decoder_layers.append(nn.Dropout(dropout))
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, embed_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        query = x @ self.rotation_params
        key = x @ self.entangle_params
        scores = F.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return scores @ x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attention(x)
        latent = self.encoder(attn_out)
        recon = self.decoder(latent)
        return recon

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.attention(x))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

def SelfAttention(embed_dim: int = 4,
                  latent_dim: int = 32,
                  hidden_dims: tuple = (128, 64),
                  dropout: float = 0.1):
    """Factory mirroring the quantum helper, returning a configured hybrid model."""
    return HybridAttentionAutoencoder(embed_dim, latent_dim, hidden_dims, dropout)

__all__ = ["SelfAttention", "HybridAttentionAutoencoder"]
