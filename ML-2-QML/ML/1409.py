import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Tuple, Iterable, Optional

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    return tensor.to(dtype=torch.float32)

@dataclass
class AutoencoderConfig:
    """Configuration for a hybrid autoencoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    dropout_schedule: Optional[Tuple[float,...]] = None
    skip: bool = True

class HybridAutoencoder(nn.Module):
    """A lightweight MLP autoencoder with optional skip connections and scheduled dropout."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        # Encoder
        encoder_layers = []
        in_dim = config.input_dim
        for i, hidden in enumerate(config.hidden_dims):
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU(inplace=True))
            if config.dropout > 0.0:
                drop = config.dropout_schedule[i] if config.dropout_schedule and i < len(config.dropout_schedule) else config.dropout
                encoder_layers.append(nn.Dropout(drop))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        # Decoder
        decoder_layers = []
        in_dim = config.latent_dim
        hidden_rev = list(reversed(config.hidden_dims))
        for i, hidden in enumerate(hidden_rev):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU(inplace=True))
            if config.dropout > 0.0:
                drop = config.dropout_schedule[-i-1] if config.dropout_schedule else config.dropout
                decoder_layers.append(nn.Dropout(drop))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        # Optional classifier head
        if config.skip:
            self.classifier = nn.Linear(config.latent_dim, config.input_dim)
        else:
            self.classifier = None

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode inputs to latent space."""
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors to reconstruction."""
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass returning reconstruction."""
        latent = self.encode(inputs)
        return self.decode(latent)

    def classify(self, inputs: torch.Tensor) -> torch.Tensor:
        """Optional classification head on latent space."""
        if self.classifier is None:
            raise RuntimeError("Classifier head not configured.")
        latent = self.encode(inputs)
        return self.classifier(latent)

__all__ = ["HybridAutoencoder", "AutoencoderConfig"]
