import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Callable, Optional

@dataclass
class AutoencoderConfig:
    """
    Configuration for the hybrid autoencoder.
    """
    input_dim: int
    latent_dim: int = 32
    hidden_dims: tuple[int, int] = (128, 64)
    dropout: float = 0.1

class HybridAutoencoderNet(nn.Module):
    """
    Hybrid classical‑quantum autoencoder.

    The encoder maps the input into a latent vector.  If a quantum encoder
    function is provided, it is applied to the latent vector before decoding.
    This allows the same architecture to be used for purely classical or
    hybrid training.
    """

    def __init__(
        self,
        config: AutoencoderConfig,
        quantum_encoder: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.quantum_encoder = quantum_encoder

        # Classical encoder
        encoder_layers = []
        in_dim = config.input_dim
        for h in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = h
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Classical decoder
        decoder_layers = []
        in_dim = config.latent_dim
        for h in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode, optionally quantum‑process, then decode."""
        latent = self.encoder(x)
        if self.quantum_encoder is not None:
            latent = self.quantum_encoder(latent)
        return self.decoder(latent)

    def set_quantum_encoder(
        self, quantum_encoder: Callable[[torch.Tensor], torch.Tensor]
    ) -> None:
        """Attach a quantum encoder function after construction."""
        self.quantum_encoder = quantum_encoder

__all__ = ["AutoencoderConfig", "HybridAutoencoderNet"]
