"""Hybrid classical autoencoder integrating a sampler decoder and kernel regularization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

# Import the sampler‑based decoder from reference 2
from SamplerQNN import SamplerQNN  # assumes same package

# Import the RBF kernel from reference 3
from QuantumKernelMethod import Kernel


def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


@dataclass
class AutoencoderConfig:
    """Configuration values for :class:`HybridAutoencoder`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    use_kernel: bool = True
    kernel_gamma: float = 1.0


class HybridAutoencoder(nn.Module):
    """A hybrid autoencoder that uses a classical encoder, a sampler‑based decoder,
    and optional RBF kernel regularisation.
    """
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config

        # Encoder
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

        # Decoder – a small sampler‑based network
        self.decoder = SamplerQNN()

        # Optional kernel regulariser
        self.kernel = Kernel(gamma=config.kernel_gamma) if config.use_kernel else None

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        latent = self.encode(inputs)
        recon = self.decode(latent)
        return recon

    def kernel_loss(self, inputs: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        """Compute an RBF kernel similarity loss between inputs and reconstructions."""
        if self.kernel is None:
            return torch.tensor(0.0, device=inputs.device)
        # Flatten batch dimension
        x = inputs.view(inputs.size(0), -1)
        y = recon.view(recon.size(0), -1)
        loss = 0.0
        for i in range(x.size(0)):
            loss += 1.0 - self.kernel(x[i], y[i])
        return loss / x.size(0)


def train_hybrid_autoencoder(
    model: HybridAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> List[float]:
    """Train the hybrid autoencoder with optional kernel loss."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse_loss = nn.MSELoss()
    history: List[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = mse_loss(recon, batch)
            if model.kernel is not None:
                loss += model.kernel_loss(batch, recon)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


__all__ = ["HybridAutoencoder", "AutoencoderConfig", "train_hybrid_autoencoder"]
