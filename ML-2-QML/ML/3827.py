"""Hybrid classical‑quantum binary classifier – classical implementation.

The network concatenates a CNN backbone, a PyTorch autoencoder (feature
reduction), and a dense hybrid head that mimics the quantum expectation
layer.  The quantum head is replaced by a sigmoid‑activated linear layer,
making the model fully classical while preserving the same interface.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Autoencoder utilities – identical to the seed's Autoencoder.py
# --------------------------------------------------------------------------- #

def _as_tensor(data):
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

class AutoencoderConfig:
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: tuple[int, int] = (128, 64),
                 dropout: float = 0.1):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

class AutoencoderNet(nn.Module):
    def __init__(self, cfg: AutoencoderConfig):
        super().__init__()
        # encoder
        enc_layers = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            enc_layers += [nn.Linear(in_dim, h), nn.ReLU(),
                           nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()]
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # decoder
        dec_layers = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            dec_layers += [nn.Linear(in_dim, h), nn.ReLU(),
                           nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()]
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))

def Autoencoder(input_dim: int, *, latent_dim: int = 32,
                hidden_dims: tuple[int, int] = (128, 64),
                dropout: float = 0.1) -> AutoencoderNet:
    cfg = AutoencoderConfig(input_dim, latent_dim, hidden_dims, dropout)
    return AutoencoderNet(cfg)

# --------------------------------------------------------------------------- #
# Hybrid head – classical sigmoid dense layer
# --------------------------------------------------------------------------- #

class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid activation that mimics the quantum expectation head."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:  # type: ignore[override]
        out = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (out,) = ctx.saved_tensors
        return grad_output * out * (1 - out), None

class Hybrid(nn.Module):
    """Dense head that replaces the quantum circuit in the original model."""
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return HybridFunction.apply(self.linear(x), self.shift)

# --------------------------------------------------------------------------- #
# Full network – convolution → autoencoder → hybrid head
# --------------------------------------------------------------------------- #

class HybridAutoencoderQCNet(nn.Module):
    """CNN backbone, autoencoder feature extractor, and dense hybrid head."""

    def __init__(self, latent_dim: int = 4, autoencoder_hidden: tuple[int, int] = (256, 128)) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Autoencoder for dimensionality reduction
        self.autoencoder = Autoencoder(
            input_dim=55815,
            latent_dim=latent_dim,
            hidden_dims=autoencoder_hidden,
            dropout=0.1,
        )

        # Classical hybrid head
        self.hybrid = Hybrid(1, shift=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Conv backbone
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)

        # Autoencoder feature extraction
        latent = self.autoencoder.encode(x)

        # Feed latent into the hybrid head
        logits = self.fc1(latent)  # simple mapping to the head input
        logits = F.relu(logits)
        logits = self.fc2(logits)
        logits = F.relu(logits)
        logits = self.fc3(logits)
        prob = self.hybrid(logits)

        return torch.cat((prob, 1 - prob), dim=-1)

__all__ = [
    "Autoencoder", "AutoencoderNet", "AutoencoderConfig",
    "HybridFunction", "Hybrid", "HybridAutoencoderQCNet",
]
