import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple

@dataclass
class AutoencoderConfig:
    """
    Configuration for the autoencoder component.
    """
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class AutoencoderNet(nn.Module):
    """
    Fully‑connected autoencoder with configurable depth and dropout.
    """
    def __init__(self, config: AutoencoderConfig) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(x))

class QCNNFeatureExtractor(nn.Module):
    """
    Classical feature extractor that mimics the QCNN convolutional and pooling stages.
    """
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 16), nn.Tanh(),
            nn.Linear(16, 16), nn.Tanh(),
            nn.Linear(16, 12), nn.Tanh(),
            nn.Linear(12, 8), nn.Tanh(),
            nn.Linear(8, 4), nn.Tanh(),
            nn.Linear(4, 4), nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class HybridQCNNAutoencoder(nn.Module):
    """
    Hybrid model that first extracts QCNN‑style features, then compresses them
    with a classical autoencoder, and finally predicts a binary label from
    the latent representation.
    """
    def __init__(
        self,
        latent_dim: int = 8,
        hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_extractor = QCNNFeatureExtractor()
        ae_cfg = AutoencoderConfig(
            input_dim=4,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        self.autoencoder = AutoencoderNet(ae_cfg)
        self.classifier = nn.Linear(latent_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.feature_extractor(x)
        encoded = self.autoencoder.encode(features)
        logits = self.classifier(encoded)
        return torch.sigmoid(logits)

def HybridQCNNAutoencoderFactory(
    latent_dim: int = 8,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> HybridQCNNAutoencoder:
    """
    Factory that returns a fully configured hybrid model.
    """
    return HybridQCNNAutoencoder(
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )

__all__ = [
    "AutoencoderConfig",
    "AutoencoderNet",
    "QCNNFeatureExtractor",
    "HybridQCNNAutoencoder",
    "HybridQCNNAutoencoderFactory",
]
