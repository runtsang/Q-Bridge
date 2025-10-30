import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple
import numpy as np

@dataclass
class AutoencoderSettings:
    """Configuration for the simple feed‑forward autoencoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class EncoderAutoNet(nn.Module):
    """Lightweight fully‑connected autoencoder used as a latent encoder."""
    def __init__(self, cfg: AutoencoderSettings) -> None:
        super().__init__()
        encoder_layers = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h))
            encoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                encoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        encoder_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h))
            decoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                decoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

def build_autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> EncoderAutoNet:
    cfg = AutoencoderSettings(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return EncoderAutoNet(cfg)

class ClassicalHybridLayer(torch.autograd.Function):
    """Differentiable sigmoid head that emulates the quantum expectation layer."""
    @staticmethod
    def forward(ctx, inp: torch.Tensor, shift: float):
        out = torch.sigmoid(inp + shift)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (out,) = ctx.saved_tensors
        return grad_out * out * (1 - out), None

class HybridAutoEncoderQCNet(nn.Module):
    """Classical binary classifier combining CNN, autoencoder latent encoder and a sigmoid head."""
    def __init__(self, latent_dim: int = 32, shift: float = 0.0) -> None:
        super().__init__()
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        # Autoencoder encoder for latent compression
        self.autoencoder = build_autoencoder(input_dim=15, latent_dim=latent_dim)
        # Classical dense head
        self.fc1 = nn.Linear(latent_dim, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)          # shape: (batch, 15)
        z = self.autoencoder.encode(x)   # latent representation
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        prob = ClassicalHybridLayer.apply(x, self.shift)
        return torch.cat((prob, 1 - prob), dim=-1)

__all__ = ["HybridAutoEncoderQCNet"]
