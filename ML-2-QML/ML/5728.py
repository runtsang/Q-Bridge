import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple

class ClassicalSelfAttention(nn.Module):
    """A lightweight self‑attention layer that mimics the structure of the quantum self‑attention block."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, embed_dim)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / (self.embed_dim ** 0.5), dim=-1)
        return torch.matmul(scores, v)

@dataclass
class AutoencoderConfig:
    """Hyper‑parameters for the hybrid autoencoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class UnifiedAutoencoder(nn.Module):
    """Classical autoencoder that integrates a self‑attention refinement block."""
    def __init__(self, cfg: AutoencoderConfig):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dims[0], cfg.hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dims[1], cfg.latent_dim),
        )
        # Self‑attention operates on the latent code
        self.attention = ClassicalSelfAttention(cfg.latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dims[1], cfg.hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dims[0], cfg.input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        # Reshape to (batch, 1, latent_dim) for the attention layer
        latent_exp = latent.unsqueeze(1)
        refined = self.attention(latent_exp).squeeze(1)
        return self.decoder(refined)
