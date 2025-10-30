"""Hybrid classical autoencoder combining quanvolution feature extraction and a stochastic sampler latent layer."""

import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple

class QuanvolutionFilter(nn.Module):
    """Classical 2×2 patch convolution inspired by quanvolution."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4, kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).view(x.size(0), -1)

class SamplerQNN(nn.Module):
    """Stochastic sampler network that maps features to a low‑dimensional latent vector."""
    def __init__(self, input_dim: int = 4 * 14 * 14, latent_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, latent_dim),
            nn.Softmax(dim=-1)
        )
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)

class HybridAutoencoder(nn.Module):
    """Full autoencoder that first extracts quanvolutional features,
    projects them with a sampler network, and then reconstructs the image."""
    def __init__(self,
                 input_shape: Tuple[int, int, int] = (1, 28, 28),
                 latent_dim: int = 32) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.encoder = QuanvolutionFilter()
        self.latent = SamplerQNN(input_dim=4 * 14 * 14, latent_dim=latent_dim)
        # Decoder mirrors the encoder: linear → conv transpose
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 4 * 14 * 14),
            nn.ReLU(),
            nn.Unflatten(1, (4, 14, 14)),
            nn.ConvTranspose2d(4, 1, kernel_size=2, stride=2)
        )
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.latent(self.encoder(x))
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)

def train_hybrid_autoencoder(model: HybridAutoencoder,
                             data: torch.Tensor,
                             *,
                             epochs: int = 50,
                             batch_size: int = 128,
                             lr: float = 1e-3,
                             device: torch.device | None = None) -> list[float]:
    """Simple training loop with MSE loss for reconstruction."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history: list[float] = []
    for _ in range(epochs):
        epoch_loss = 0.0
        for batch, in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

__all__ = ["HybridAutoencoder", "SamplerQNN", "QuanvolutionFilter", "train_hybrid_autoencoder"]
