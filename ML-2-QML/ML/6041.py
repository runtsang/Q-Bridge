import torch
import torch.nn as nn
from typing import Tuple

class AutoencoderConfig:
    """Configuration for the dense auto‑encoder."""
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: Tuple[int, int] = (128, 64), dropout: float = 0.1):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

class AutoencoderNet(nn.Module):
    """Fully‑connected auto‑encoder."""
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

class QuanvolutionAutoencoder(nn.Module):
    """Classical hybrid of a 2‑D quanvolution filter and a dense auto‑encoder."""
    def __init__(self,
                 in_channels: int = 1,
                 conv_out_channels: int = 4,
                 conv_kernel: int = 2,
                 conv_stride: int = 2,
                 autoencoder_config: AutoencoderConfig | None = None,
                 image_size: int = 28) -> None:
        super().__init__()
        self.qfilter = nn.Conv2d(in_channels, conv_out_channels,
                                 kernel_size=conv_kernel,
                                 stride=conv_stride,
                                 bias=False)
        conv_dim = image_size // conv_stride
        flat_dim = conv_out_channels * conv_dim * conv_dim
        if autoencoder_config is None:
            autoencoder_config = AutoencoderConfig(input_dim=flat_dim)
        self.autoencoder = AutoencoderNet(autoencoder_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        flat = features.view(features.size(0), -1)
        recon = self.autoencoder(flat)
        return recon.view(-1, 1, 28, 28)

__all__ = ["AutoencoderConfig", "AutoencoderNet", "QuanvolutionAutoencoder"]
