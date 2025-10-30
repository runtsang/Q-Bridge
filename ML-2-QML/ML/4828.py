from __future__ import annotations

import torch
from torch import nn
from typing import Iterable, Optional

from.Autoencoder import AutoencoderConfig, AutoencoderNet

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


class HybridAutoencoderNet(nn.Module):
    """Hybrid autoencoder that optionally augments the latent vector
    with a quantum estimator network."""
    def __init__(self, config: AutoencoderConfig,
                 quantum_estimator: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.autoencoder = AutoencoderNet(config)
        self.quantum_estimator = quantum_estimator

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        latent = self.autoencoder.encode(inputs)
        if self.quantum_estimator is not None:
            latent = self.quantum_estimator(latent)
        return latent

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.autoencoder.decode(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(inputs))


def HybridAutoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    quantum_estimator: Optional[nn.Module] = None,
) -> HybridAutoencoderNet:
    """Factory that wires a classical autoencoder with an optional
    quantum estimator."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return HybridAutoencoderNet(config, quantum_estimator=quantum_estimator)


def train_hybrid_autoencoder(
    model: HybridAutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Reconstruction training loop that accepts a hybrid encoder/decoder."""
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


__all__ = ["HybridAutoencoder", "HybridAutoencoderNet", "train_hybrid_autoencoder"]
