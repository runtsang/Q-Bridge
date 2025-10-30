"""
Hybrid classical module combining a fully‑connected autoencoder with a linear head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Import the autoencoder utilities from the seed
from Autoencoder import AutoencoderNet, AutoencoderConfig, train_autoencoder

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
class HybridFCLConfig:
    """Configuration for the hybrid fully‑connected layer."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    fc_output_dim: int = 1  # output dimension of the linear head


class HybridFCL(nn.Module):
    """Hybrid module: autoencoder encoder + linear fully‑connected head."""

    def __init__(self, config: HybridFCLConfig) -> None:
        super().__init__()
        self.autoencoder = AutoencoderNet(
            AutoencoderConfig(
                input_dim=config.input_dim,
                latent_dim=config.latent_dim,
                hidden_dims=config.hidden_dims,
                dropout=config.dropout,
            )
        )
        self.fc = nn.Linear(config.latent_dim, config.fc_output_dim)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.encode(inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        latent = self.encode(inputs)
        return self.fc(latent)

    def run(self, thetas: Iterable[float]) -> torch.Tensor:
        """
        Set the linear head weights from a flat list of parameters and
        evaluate on a dummy zero input.  Useful for quick numerical checks.
        """
        weight = torch.tensor(
            thetas[: self.fc.weight.numel()], dtype=torch.float32
        ).view_as(self.fc.weight)
        bias = torch.tensor(
            thetas[self.fc.weight.numel() :], dtype=torch.float32
        ).view_as(self.fc.bias)
        self.fc.weight.data = weight
        self.fc.bias.data = bias
        dummy = torch.zeros(
            self.autoencoder.encoder[0].in_features, dtype=torch.float32
        ).unsqueeze(0)
        return self.forward(dummy)


def HybridFCLFactory(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    fc_output_dim: int = 1,
) -> HybridFCL:
    """Convenience factory mirroring the original FCL helper."""
    config = HybridFCLConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        fc_output_dim=fc_output_dim,
    )
    return HybridFCL(config)


def train_hybrid(
    model: HybridFCL,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """
    Simple reconstruction training loop that optimises both the autoencoder
    and the linear head simultaneously.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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


__all__ = [
    "HybridFCL",
    "HybridFCLFactory",
    "train_hybrid",
    "HybridFCLConfig",
]
