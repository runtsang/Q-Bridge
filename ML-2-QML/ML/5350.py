import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable

class SelfAttentionModule(nn.Module):
    """Classical self‑attention block mimicking the quantum interface."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        # Parameter matrices shaped like rotation and entangle parameters
        self.rotation = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.entangle = nn.Parameter(torch.randn(embed_dim, embed_dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (batch, embed_dim)
        query = inputs @ self.rotation
        key = inputs @ self.entangle
        scores = F.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return scores @ inputs

class AutoencoderConfig:
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: tuple[int, int] = (128, 64), dropout: float = 0.1):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

class AutoencoderNet(nn.Module):
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(inputs))

class UnifiedSelfAttentionAutoEncoderHybrid(nn.Module):
    """
    Classical hybrid module that combines self‑attention, a
    variational auto‑encoder, and an optional quantum head.
    """
    def __init__(
        self,
        embed_dim: int = 4,
        latent_dim: int = 32,
        hidden_dims: tuple[int, int] = (128, 64),
        dropout: float = 0.1,
        quantum_head: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__()
        self.self_attention = SelfAttentionModule(embed_dim)
        self.autoencoder = AutoencoderNet(
            AutoencoderConfig(
                input_dim=embed_dim,
                latent_dim=latent_dim,
                hidden_dims=hidden_dims,
                dropout=dropout,
            )
        )
        self.quantum_head = quantum_head

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: self‑attention → auto‑encoder → (optional) quantum head.
        The output is a probability pair for binary classification.
        """
        # Self‑attention
        attn_out = self.self_attention(inputs)
        # Auto‑encoder latent representation
        latent = self.autoencoder.encode(attn_out)
        recon = self.autoencoder.decode(latent)
        # Quantum head if provided
        if self.quantum_head is not None:
            logits = self.quantum_head(recon)
        else:
            # Default linear map to a single logit
            logits = nn.functional.linear(recon, torch.ones_like(recon))
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["UnifiedSelfAttentionAutoEncoderHybrid"]
