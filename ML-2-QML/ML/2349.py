"""Hybrid classical autoencoder with a quantum‑inspired fully‑connected layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------------------------------- #
# 1. Quantum‑inspired fully‑connected layer
# --------------------------------------------------------------------------- #
class QuantumInspiredFCL(nn.Module):
    """
    Mimics a single‑qubit variational layer using a tanh activation.
    Parameters are treated as rotation angles; the layer outputs the
    mean expectation value across a batch of angles.
    """
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        # thetas: shape (batch, n_features)
        expectation = torch.tanh(self.linear(thetas)).mean(dim=0, keepdim=True)
        return expectation

# --------------------------------------------------------------------------- #
# 2. Configuration dataclass
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderHybridConfig:
    """Hyper‑parameters for the hybrid autoencoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    quantum_layer_dim: int = 1  # number of parameters for the quantum‑inspired layer

# --------------------------------------------------------------------------- #
# 3. Hybrid autoencoder network
# --------------------------------------------------------------------------- #
class AutoencoderHybrid(nn.Module):
    """
    Fully‑connected autoencoder that inserts a quantum‑inspired layer
    at the bottleneck. The layer is a lightweight surrogate for a
    single‑qubit variational circuit.
    """
    def __init__(self, config: AutoencoderHybridConfig) -> None:
        super().__init__()
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()
            ])
            in_dim = hidden
        # Bottleneck: linear → quantum‑inspired layer
        encoder_layers.append(nn.Linear(in_dim, config.quantum_layer_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        self.quantum_fcl = QuantumInspiredFCL(config.quantum_layer_dim)

        decoder_layers = []
        in_dim = config.quantum_layer_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.extend([
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()
            ])
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(inputs)
        return self.quantum_fcl(latent)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(inputs))

# --------------------------------------------------------------------------- #
# 4. Factory and training helper
# --------------------------------------------------------------------------- #
def make_autoencoder_hybrid(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    quantum_layer_dim: int = 1,
) -> AutoencoderHybrid:
    """Return a configured hybrid autoencoder."""
    config = AutoencoderHybridConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        quantum_layer_dim=quantum_layer_dim,
    )
    return AutoencoderHybrid(config)

def train_autoencoder_hybrid(
    model: AutoencoderHybrid,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Simple reconstruction training loop returning the loss history."""
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

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Utility to ensure a float32 tensor on the default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

__all__ = [
    "AutoencoderHybrid",
    "AutoencoderHybridConfig",
    "QuantumInspiredFCL",
    "train_autoencoder_hybrid",
    "make_autoencoder_hybrid",
]
