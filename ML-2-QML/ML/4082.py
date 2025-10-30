import torch
from torch import nn
import numpy as np

class HybridConvFilter(nn.Module):
    """
    Classical convolutional filter with optional autoencoder reconstruction.
    The filter can be used as a drop‑in replacement for the original Conv.py.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0,
                 dropout: float = 0.0, latent_dim: int = 32):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        self.dropout = nn.Dropout(dropout)

        # Encoder to a latent space
        self.encoder = nn.Linear(kernel_size * kernel_size, latent_dim)
        # Decoder to reconstruct the input patch
        self.decoder = nn.Linear(latent_dim, kernel_size * kernel_size)

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch, 1, kernel_size, kernel_size)

        Returns:
            activations: Tensor of activations after sigmoid and threshold.
            reconstructions: Tensor of reconstructed patches.
        """
        conv_out = self.conv(x)
        act = torch.sigmoid(conv_out - self.threshold)
        act = self.dropout(act)

        flat = act.view(act.size(0), -1)
        latent = self.encoder(flat)
        recon_flat = self.decoder(latent)
        recon = recon_flat.view(act.shape)
        return act, recon

def Conv(**kwargs):
    """
    Factory that returns a HybridConvFilter instance.
    """
    return HybridConvFilter(**kwargs)
