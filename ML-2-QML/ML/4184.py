"""Classical implementation of a hybrid binary classifier that integrates
a lightweight auto‑encoder for feature compression and a sampler‑style
neural head for probability estimation.  It mirrors the structure of the
original quantum model while staying fully differentiable in PyTorch."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple

@dataclass
class AutoencoderConfig:
    """Configuration for the classical auto‑encoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class AutoencoderNet(nn.Module):
    """Multilayer perceptron auto‑encoder."""
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        encoder_layers = []
        in_dim = cfg.input_dim
        for hidden in cfg.hidden_dims:
            encoder_layers += [nn.Linear(in_dim, hidden), nn.ReLU()]
            if cfg.dropout > 0:
                encoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = cfg.latent_dim
        for hidden in reversed(cfg.hidden_dims):
            decoder_layers += [nn.Linear(in_dim, hidden), nn.ReLU()]
            if cfg.dropout > 0:
                decoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

class SamplerNet(nn.Module):
    """A lightweight softmax head that emulates the quantum SamplerQNN."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(logits), dim=-1)

class HybridQCNet(nn.Module):
    """Classical hybrid classifier that combines a feature auto‑encoder
    with a sampler‑style probability head."""
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 32,
                 hidden_dims: Tuple[int, int] = (128, 64),
                 dropout: float = 0.1) -> None:
        super().__init__()
        cfg = AutoencoderConfig(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        self.autoencoder = AutoencoderNet(cfg)
        self.classifier = nn.Linear(latent_dim, 1)
        self.sampler = SamplerNet()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.autoencoder.encode(x)
        logits = self.classifier(z).view(-1, 1)
        probs = self.sampler(logits)
        return probs

__all__ = ["HybridQCNet"]
