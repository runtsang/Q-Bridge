"""Hybrid classical-quantum autoencoder combining a classical MLP encoder/decoder with
a quantum kernel layer and optional SamplerQNN for latent feature enhancement."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Callable, Optional, List, Iterable

import torch
from torch import nn
import torch.nn.functional as F

@dataclass
class HybridAutoencoderConfig:
    """Configuration values for :class:`HybridAutoencoderNet`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class HybridAutoencoderNet(nn.Module):
    """Classical encoder + optional quantum layer + classical decoder."""
    def __init__(
        self,
        config: HybridAutoencoderConfig,
        quantum_layer: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        kernel_regularizer: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> None:
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

        # Optional quantum transformation
        self.quantum_layer = quantum_layer

        # Decoder
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

        # Kernel regularizer
        self.kernel_regularizer = kernel_regularizer

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def quantum_transform(self, latent: torch.Tensor) -> torch.Tensor:
        if self.quantum_layer is None:
            return latent
        return self.quantum_layer(latent)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        latent = self.encode(inputs)
        latent_q = self.quantum_transform(latent)
        return self.decode(latent_q)

    def kernel_loss(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.kernel_regularizer is None:
            return torch.tensor(0.0, device=inputs.device)
        latent = self.encode(inputs)
        K = self.kernel_regularizer(latent, latent)
        # Encourage orthogonality: trace(K) - batch_size
        return (K.diag().sum() - latent.size(0)).float().mean()

def HybridAutoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    quantum_layer: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    kernel_regularizer: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
) -> HybridAutoencoderNet:
    """Factory mirroring the classical helper but with quantum hooks."""
    config = HybridAutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return HybridAutoencoderNet(config, quantum_layer, kernel_regularizer)

def train_hybrid_autoencoder(
    model: HybridAutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
    lambda_kernel: float = 0.0,
) -> List[float]:
    """Training loop that optionally adds kernel regularization."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(_as_tensor(data))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            if lambda_kernel > 0.0:
                k_loss = model.kernel_loss(batch)
                loss += lambda_kernel * k_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

__all__ = [
    "HybridAutoencoder",
    "HybridAutoencoderConfig",
    "HybridAutoencoderNet",
    "train_hybrid_autoencoder",
]
