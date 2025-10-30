import torch
import torch.nn as nn
from typing import Any, Tuple

class AutoencoderConfig:
    """Configuration for the classical autoencoder."""
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: Tuple[int, int] = (128, 64), dropout: float = 0.1):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

class AutoencoderNet(nn.Module):
    """Fully‑connected autoencoder mirroring the reference implementation."""
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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

class HybridAutoSampler(nn.Module):
    """
    Classical core that delegates decoding to a quantum sampler.
    The encoder produces a latent vector that is interpreted as
    parameters for the quantum circuit.  The quantum sampler returns
    a probability distribution treated as the reconstructed output.
    """
    def __init__(self, config: AutoencoderConfig, qml_sampler: Any):
        super().__init__()
        self.encoder = AutoencoderNet(config)
        self.qml_sampler = qml_sampler

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder.encode(x)
        samples = self.qml_sampler.sample(latent)
        return torch.tensor(samples, dtype=torch.float32)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.tensor(self.qml_sampler.sample(z), dtype=torch.float32)
