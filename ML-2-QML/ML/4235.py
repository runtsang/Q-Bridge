"""Hybrid classical‑quantum autoencoder integrating QCNN‑style encoder and variational quantum latent representation."""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple, Callable

@dataclass
class AutoencoderConfig:
    """Configuration for the hybrid autoencoder."""
    input_shape: Tuple[int, int, int]  # (channels, height, width)
    latent_dim: int = 16
    hidden_dims: Tuple[int, int] = (256, 128)
    dropout: float = 0.1
    quantum_encoder: Callable[..., nn.Module] | None = None

class HybridAutoencoder(nn.Module):
    """Classical‑quantum autoencoder.

    The encoder consists of a QCNN‑inspired convolutional stack that maps the input image to a
    low‑dimensional feature vector. This vector is then fed to a variational quantum circuit
    (supplied via ``quantum_encoder``) that produces the latent representation. The decoder
    is a fully‑connected MLP that reconstructs the input from the latent vector.
    """

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config

        # QCNN‑style encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(config.input_shape[0], 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        conv_out_dim = self._conv_output_dim(config.input_shape)
        self.fc_to_qubits = nn.Linear(conv_out_dim, 8)  # map to 8‑qubit feature vector

        # Quantum encoder
        if config.quantum_encoder is None:
            raise ValueError("quantum_encoder must be supplied")
        self.quantum_encoder = config.quantum_encoder(
            num_qubits=8,
            latent_dim=config.latent_dim
        )

        # Decoder MLP
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dims[0], config.hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dims[1], config.input_shape[0] * config.input_shape[1] * config.input_shape[2]),
            nn.Sigmoid()
        )

    def _conv_output_dim(self, shape: Tuple[int, int, int]) -> int:
        """Compute the flattened size after the QCNN encoder."""
        dummy = torch.zeros(1, *shape)
        out = self.encoder_conv(dummy)
        return out.view(1, -1).size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)
        features = self.fc_to_qubits(x)
        latent = self.quantum_encoder(features)
        recon = self.decoder(latent)
        recon = recon.view(x.size(0), *self.config.input_shape)
        return recon

def train_autoencoder(
    model: nn.Module,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Training routine returning a list of MSE loss values per epoch."""
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
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

def _as_tensor(data: torch.Tensor | Iterable[float]) -> torch.Tensor:
    """Utility converting input to float32 tensor on the default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

__all__ = [
    "AutoencoderConfig",
    "HybridAutoencoder",
    "train_autoencoder",
]
