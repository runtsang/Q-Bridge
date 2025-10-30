import numpy as np
import torch
from torch import nn
from typing import Tuple

class ConvFilter(nn.Module):
    """Classical 2‑D convolution filter that emulates a quanvolution layer."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.conv(x)
        return torch.sigmoid(logits - self.threshold)

class AutoencoderConfig:
    """Configuration for :class:`AutoencoderNet`."""
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 32,
                 hidden_dims: Tuple[int, int] = (128, 64),
                 dropout: float = 0.1) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

class AutoencoderNet(nn.Module):
    """Fully‑connected autoencoder with configurable hidden layers."""
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

class ConvAutoencoder(nn.Module):
    """
    Hybrid classical pipeline:
        1. Extract patches with a convolution filter.
        2. Compress / reconstruct each patch with a fully‑connected autoencoder.
    """
    def __init__(self,
                 conv_kernel: int = 2,
                 conv_threshold: float = 0.0,
                 latent_dim: int = 32,
                 hidden_dims: Tuple[int, int] = (128, 64),
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.conv_filter = ConvFilter(kernel_size=conv_kernel, threshold=conv_threshold)
        patch_size = conv_kernel * conv_kernel
        config = AutoencoderConfig(input_dim=patch_size,
                                   latent_dim=latent_dim,
                                   hidden_dims=hidden_dims,
                                   dropout=dropout)
        self.autoencoder = AutoencoderNet(config)
        self.unfold = nn.Unfold(kernel_size=conv_kernel, stride=1, padding=0)
        self.fold = nn.Fold(output_size=None)  # will be set on first forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (batch, 1, H, W).

        Returns
        -------
        torch.Tensor
            Reconstructed image of the same shape.
        """
        # Extract patches
        patches = self.unfold(x)  # (batch, k*k, L)
        batch, _, L = patches.shape
        patches = patches.permute(0, 2, 1).reshape(batch * L, -1)  # (batch*L, k*k)

        # Encode & decode patches
        recon = self.autoencoder(patches)  # (batch*L, k*k)
        recon = recon.reshape(batch, L, -1).permute(0, 2, 1)  # (batch, k*k, L)

        # Fold patches back into image
        if self.fold.output_size is None:
            _, _, H, W = x.shape
            self.fold.output_size = (H, W)
        recon_img = self.fold(recon)
        return recon_img

    def run(self, data: np.ndarray) -> np.ndarray:
        """
        Convenience wrapper that accepts a NumPy array.

        Parameters
        ----------
        data : np.ndarray
            Grayscale image of shape (H, W).

        Returns
        -------
        np.ndarray
            Reconstructed image of the same shape.
        """
        tensor = torch.as_tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        with torch.no_grad():
            out = self.forward(tensor)
        return out.squeeze().numpy()

__all__ = ["ConvAutoencoder"]
