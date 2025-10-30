import torch
from torch import nn
from typing import Sequence

class ConvFilter(nn.Module):
    """Classical 2‑D convolution filter with sigmoid activation."""
    def __init__(self, kernel_size: int = 3, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=[2, 3])  # (batch, 1)

class AutoencoderNet(nn.Module):
    """Fully‑connected autoencoder with configurable depth."""
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: Sequence[int] = (128, 64), dropout: float = 0.1) -> None:
        super().__init__()
        enc_layers = []
        in_dim = input_dim
        for h in hidden_dims:
            enc_layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            if dropout > 0:
                enc_layers.append(nn.Dropout(dropout))
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            dec_layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            if dropout > 0:
                dec_layers.append(nn.Dropout(dropout))
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

class RBFKernel(nn.Module):
    """Radial‑basis function kernel."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * (diff * diff).sum(dim=-1, keepdim=True))

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor],
                  gamma: float = 1.0) -> torch.Tensor:
    k = RBFKernel(gamma)
    return torch.stack([k(a[i], b) for i in range(len(a))]).squeeze()

class SharedClassName(nn.Module):
    """Hybrid classical pipeline: convolution ➜ autoencoder ➜ kernel similarity."""
    def __init__(self, conv_k: int = 3, latent: int = 16,
                 hidden: Sequence[int] = (64, 32), gamma: float = 0.5) -> None:
        super().__init__()
        self.conv = ConvFilter(kernel_size=conv_k, threshold=0.0)
        self.auto = AutoencoderNet(input_dim=conv_k * conv_k,
                                   latent_dim=latent,
                                   hidden_dims=hidden)
        self.kernel = RBFKernel(gamma=gamma)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (batch, 1, H, W).

        Returns
        -------
        recon : torch.Tensor
            Reconstructed image from the autoencoder.
        sims : torch.Tensor
            Pairwise similarity matrix of latent codes.
        """
        conv_out = self.conv(x)                      # (batch, 1)
        flat = conv_out.view(conv_out.size(0), -1)    # flatten
        z = self.auto.encode(flat)
        recon = self.auto.decode(z)
        sims = self.kernel(z, z)
        return recon, sims

__all__ = ["SharedClassName"]
