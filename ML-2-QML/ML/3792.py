"""Hybrid Autoencoder combining classical quanvolution, quantum variational autoencoder, and classical decoder."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple

# Import the quantum latent layer
from.quantum_autoencoder import QuantumAutoencoder


@dataclass
class AutoencoderConfig:
    """Configuration values for :class:`HybridAutoencoderNet`."""
    input_dim: int  # for flattened input; e.g., 28*28=784
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    num_trash: int = 2
    reps: int = 5


class QuanvolutionFilter(nn.Module):
    """Classical 2×2 patch convolution inspired by the quanvolution example."""
    def __init__(self) -> None:
        super().__init__()
        # 1 input channel → 4 feature maps
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.conv(x)
        return features.view(x.size(0), -1)  # flatten per sample


class HybridAutoencoderNet(nn.Module):
    """
    A hybrid autoencoder that:
    * encodes images with a classical quanvolution filter + linear map to a latent vector,
    * transforms the latent vector with a quantum variational circuit (QuantumAutoencoder),
    * decodes the transformed latent back to the image domain via a transposed convolution.
    """
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Classical encoder: quanvolution + linear projection
        self.encoder = nn.Sequential(
            QuanvolutionFilter(),
            nn.Flatten(),
            nn.Linear(4 * 14 * 14, cfg.latent_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout)
        )

        # Quantum latent transformation
        self.quantum = QuantumAutoencoder(
            latent_dim=cfg.latent_dim,
            num_trash=cfg.num_trash,
            reps=cfg.reps
        )

        # Classical decoder: linear expansion + transposed convolution
        self.decoder = nn.Sequential(
            nn.Linear(cfg.latent_dim, 4 * 14 * 14),
            nn.ReLU(),
            nn.Unflatten(1, (4, 14, 14)),
            nn.ConvTranspose2d(4, 1, kernel_size=2, stride=2)
        )

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode an image to a transformed latent representation."""
        latent = self.encoder(inputs)
        transformed = self.quantum.encode(latent)
        return transformed

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode a transformed latent back to an image."""
        decoded_latent = self.quantum.decode(latents)
        reconstruction = self.decoder(decoded_latent)
        return reconstruction

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))


def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    num_trash: int = 2,
    reps: int = 5
) -> HybridAutoencoderNet:
    """Factory returning a fully configured hybrid autoencoder."""
    cfg = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        num_trash=num_trash,
        reps=reps
    )
    return HybridAutoencoderNet(cfg)


def train_autoencoder(
    model: HybridAutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None
) -> list[float]:
    """Train the hybrid autoencoder and return the loss history."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(_as_tensor(data))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


def _as_tensor(data: torch.Tensor | list[float] | tuple[float,...]) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


__all__ = [
    "Autoencoder",
    "AutoencoderConfig",
    "HybridAutoencoderNet",
    "train_autoencoder",
]
