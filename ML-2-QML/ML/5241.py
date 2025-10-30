"""Hybrid sampler autoencoder combining classical autoencoder, transformer, and fraud‑detection inspired layers."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Iterable

# ----------------------------------------------------------------------
# FraudDetection‑inspired layer
# ----------------------------------------------------------------------
@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

class FraudLayer(nn.Module):
    """Linear + tanh + scale/shift layer with optional clipping."""
    def __init__(self, params: FraudLayerParameters, clip: bool = False):
        super().__init__()
        weight = torch.tensor(
            [[params.bs_theta, params.bs_phi],
             [params.squeeze_r[0], params.squeeze_r[1]]],
            dtype=torch.float32
        )
        bias = torch.tensor(params.phases, dtype=torch.float32)
        if clip:
            weight = weight.clamp(-5.0, 5.0)
            bias = bias.clamp(-5.0, 5.0)
        self.linear = nn.Linear(2, 2)
        with torch.no_grad():
            self.linear.weight.copy_(weight)
            self.linear.bias.copy_(bias)
        self.activation = nn.Tanh()
        self.scale = nn.Parameter(torch.tensor(params.displacement_r, dtype=torch.float32))
        self.shift = nn.Parameter(torch.tensor(params.displacement_phi, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(self.linear(x))
        out = out * self.scale + self.shift
        return out

# ----------------------------------------------------------------------
# Autoencoder
# ----------------------------------------------------------------------
@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class AutoencoderNet(nn.Module):
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        # First layer uses FraudLayer
        self.input_layer = FraudLayer(
            FraudLayerParameters(
                bs_theta=0.0, bs_phi=0.0,
                phases=(0.0, 0.0),
                squeeze_r=(1.0, 1.0),
                squeeze_phi=(0.0, 0.0),
                displacement_r=(1.0, 1.0),
                displacement_phi=(0.0, 0.0),
                kerr=(0.0, 0.0)
            ),
            clip=False
        )
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
        x = self.input_layer(inputs)
        return self.encoder(x)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(inputs))

# ----------------------------------------------------------------------
# Minimal transformer block (classical)
# ----------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

# ----------------------------------------------------------------------
# Sampler module
# ----------------------------------------------------------------------
class SamplerMLModule(nn.Module):
    def __init__(self, latent_dim: int = 32, hidden_dim: int = 64,
                 transformer_heads: int = 4, transformer_ffn: int = 128):
        super().__init__()
        self.transformer = TransformerBlock(latent_dim, transformer_heads, transformer_ffn)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Treat latent vector as sequence length 1
        seq = z.unsqueeze(1)
        seq = self.transformer(seq)
        seq = seq.squeeze(1)
        return F.softmax(self.mlp(seq), dim=-1)

# ----------------------------------------------------------------------
# Combined hybrid model
# ----------------------------------------------------------------------
class HybridSamplerAutoEncoder(nn.Module):
    """Combines an autoencoder with a transformer‑based sampler."""
    def __init__(self,
                 input_dim: int = 2,
                 latent_dim: int = 32,
                 hidden_dims: Tuple[int, int] = (128, 64),
                 dropout: float = 0.1,
                 sampler_hidden_dim: int = 64,
                 transformer_heads: int = 4,
                 transformer_ffn: int = 128):
        super().__init__()
        config = AutoencoderConfig(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout
        )
        self.autoencoder = AutoencoderNet(config)
        self.sampler = SamplerMLModule(
            latent_dim=latent_dim,
            hidden_dim=sampler_hidden_dim,
            transformer_heads=transformer_heads,
            transformer_ffn=transformer_ffn
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.autoencoder.encode(x)
        return self.sampler(z)

__all__ = [
    "FraudLayerParameters",
    "FraudLayer",
    "AutoencoderConfig",
    "AutoencoderNet",
    "TransformerBlock",
    "SamplerMLModule",
    "HybridSamplerAutoEncoder",
]
