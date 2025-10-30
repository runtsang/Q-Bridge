import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple

@dataclass
class HybridAutoencoderConfig:
    """Configuration for the hybrid autoencoder."""
    input_dim: int
    latent_dim: int = 32
    dropout: float = 0.1
    use_tanh: bool = False

class HybridAutoencoder(nn.Module):
    """Classical hybrid autoencoder that mimics a QCNN encoder followed by a dense decoder."""
    def __init__(self, config: HybridAutoencoderConfig) -> None:
        super().__init__()
        # Encoder: QCNN-inspired layers
        encoder_layers = []
        # Feature map
        encoder_layers.append(nn.Linear(config.input_dim, 16))
        encoder_layers.append(nn.Tanh() if config.use_tanh else nn.ReLU())
        if config.dropout > 0.0:
            encoder_layers.append(nn.Dropout(config.dropout))
        # Convolutional layer
        encoder_layers.append(nn.Linear(16, 16))
        encoder_layers.append(nn.Tanh() if config.use_tanh else nn.ReLU())
        if config.dropout > 0.0:
            encoder_layers.append(nn.Dropout(config.dropout))
        # Pooling layer
        encoder_layers.append(nn.Linear(16, 12))
        encoder_layers.append(nn.Tanh() if config.use_tanh else nn.ReLU())
        if config.dropout > 0.0:
            encoder_layers.append(nn.Dropout(config.dropout))
        # Convolutional layer
        encoder_layers.append(nn.Linear(12, 8))
        encoder_layers.append(nn.Tanh() if config.use_tanh else nn.ReLU())
        if config.dropout > 0.0:
            encoder_layers.append(nn.Dropout(config.dropout))
        # Pooling layer
        encoder_layers.append(nn.Linear(8, 4))
        encoder_layers.append(nn.Tanh() if config.use_tanh else nn.ReLU())
        if config.dropout > 0.0:
            encoder_layers.append(nn.Dropout(config.dropout))
        # Final convolution
        encoder_layers.append(nn.Linear(4, 4))
        encoder_layers.append(nn.Tanh() if config.use_tanh else nn.ReLU())
        if config.dropout > 0.0:
            encoder_layers.append(nn.Dropout(config.dropout))
        # Latent layer
        encoder_layers.append(nn.Linear(4, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder: simple fully‑connected reconstruction
        decoder_layers = [
            nn.Linear(config.latent_dim, 4),
            nn.ReLU(),
            nn.Linear(4, config.input_dim),
        ]
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

def HybridAutoencoderFactory(
    input_dim: int,
    *,
    latent_dim: int = 32,
    dropout: float = 0.1,
    use_tanh: bool = False,
) -> HybridAutoencoder:
    """Instantiate a hybrid autoencoder with the specified configuration."""
    config = HybridAutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        dropout=dropout,
        use_tanh=use_tanh,
    )
    return HybridAutoencoder(config)

__all__ = ["HybridAutoencoder", "HybridAutoencoderFactory", "HybridAutoencoderConfig"]
