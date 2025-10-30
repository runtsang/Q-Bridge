import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple

# ----------------------------------------------------------------------
# Autoencoder components (inspired by Autoencoder.py)
# ----------------------------------------------------------------------
@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

def Autoencoder(input_dim: int, *, latent_dim: int = 32, hidden_dims: Tuple[int, int] = (128, 64), dropout: float = 0.1) -> AutoencoderNet:
    config = AutoencoderConfig(input_dim=input_dim, latent_dim=latent_dim, hidden_dims=hidden_dims, dropout=dropout)
    return AutoencoderNet(config)

# ----------------------------------------------------------------------
# Conv filter (inspired by Conv.py)
# ----------------------------------------------------------------------
class ConvFilter(nn.Module):
    """
    2‑D convolutional feature extractor that produces a 2‑dimensional feature
    vector for each sample.  The two output channels are averaged over spatial
    dimensions to match the 2‑feature input expected by the autoencoder.
    """
    def __init__(self, kernel_size: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, H, W).

        Returns
        -------
        torch.Tensor
            Feature tensor of shape (batch, 2).
        """
        out = self.conv(x)                     # (batch, 2, H‑k+1, W‑k+1)
        out = out.mean(dim=(2, 3))             # average spatially → (batch, 2)
        return out

# ----------------------------------------------------------------------
# FraudDetectionHybrid (classical)
# ----------------------------------------------------------------------
class FraudDetectionHybrid(nn.Module):
    """
    Hybrid classical model that stitches together a convolutional feature extractor,
    a lightweight autoencoder, and a shallow classifier.  It is inspired by the
    photonic FraudDetection circuit, the EstimatorQNN regressor, the autoencoder
    architecture, and the quanvolution filter.
    """
    def __init__(
        self,
        conv_kernel: int = 2,
        autoencoder_config: AutoencoderConfig | None = None,
        classifier_hidden: Tuple[int, int] = (32, 16),
    ) -> None:
        super().__init__()
        self.conv = ConvFilter(kernel_size=conv_kernel)
        # Autoencoder expects 2‑dimensional input; the conv filter outputs 2 features.
        config = autoencoder_config or AutoencoderConfig(
            input_dim=2, latent_dim=2, hidden_dims=(8, 4), dropout=0.0
        )
        self.autoencoder = AutoencoderNet(config)

        self.classifier = nn.Sequential(
            nn.Linear(config.latent_dim, classifier_hidden[0]),
            nn.ReLU(),
            nn.Linear(classifier_hidden[0], classifier_hidden[1]),
            nn.ReLU(),
            nn.Linear(classifier_hidden[1], 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (batch, 1, H, W).

        Returns
        -------
        torch.Tensor
            Predicted fraud probability of shape (batch, 1).
        """
        features = self.conv(x)                 # (batch, 2)
        latent = self.autoencoder.encode(features)  # (batch, latent_dim)
        out = self.classifier(latent)           # (batch, 1)
        return out

__all__ = ["Autoencoder", "AutoencoderConfig", "AutoencoderNet", "ConvFilter", "FraudDetectionHybrid"]
