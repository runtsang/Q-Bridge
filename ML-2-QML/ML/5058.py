import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple

# ---------- Autoencoder utilities ----------
class AutoencoderConfig:
    """Configuration for the autoencoder."""
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: Tuple[int, int] = (128, 64), dropout: float = 0.1):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

class AutoencoderNet(nn.Module):
    """Multilayer perceptron autoencoder."""
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        # Encoder
        enc_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            enc_layers.append(nn.Linear(in_dim, hidden))
            enc_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                enc_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        enc_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)
        # Decoder
        dec_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            dec_layers.append(nn.Linear(in_dim, hidden))
            dec_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                dec_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        dec_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

# ---------- Hybrid model ----------
class QuanvolutionHybrid(nn.Module):
    """
    Classical hybrid model that combines:
      * 2‑D convolution (quanvolution style)
      * Classical autoencoder for feature compression
      * Small regression head
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 8,
        kernel_size: int = 2,
        stride: int = 2,
        latent_dim: int = 32,
        autoencoder_hidden: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
        regression_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        # Determine output dimensionality
        dummy = torch.zeros(1, in_channels, 28, 28)
        conv_out = self.conv(dummy).view(1, -1)
        self.conv_output_dim = conv_out.shape[1]
        # Autoencoder
        ae_cfg = AutoencoderConfig(
            input_dim=self.conv_output_dim,
            latent_dim=latent_dim,
            hidden_dims=autoencoder_hidden,
            dropout=dropout
        )
        self.autoencoder = AutoencoderNet(ae_cfg)
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(latent_dim, regression_hidden),
            nn.ReLU(),
            nn.Linear(regression_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, 1, 28, 28) – compatible with MNIST‑style data.

        Returns
        -------
        torch.Tensor
            Log‑softmax over 10 classes or a scalar regression output.
        """
        features = self.conv(x)
        flat = features.view(features.size(0), -1)
        compressed = self.autoencoder.encode(flat)
        out = self.regressor(compressed)
        return F.log_softmax(out, dim=-1) if out.shape[-1] > 1 else out

__all__ = ["QuanvolutionHybrid"]
