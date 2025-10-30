"""Hybrid auto‑encoder combining classical and quantum processing.

The network follows:
    input → ConvFilter → Flatten → FC → QuantumLayer → FC → Decoder → output
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Quantum wrapper (see quantum_autoencoder module)
from.quantum_autoencoder import HybridQNN


# ----------------------------------------------------------------------
# Classical helpers (adapted from the reference Conv and FCL modules)
# ----------------------------------------------------------------------
class ConvFilter(nn.Module):
    """2‑D convolution followed by a sigmoid threshold."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations


class FullyConnectedLayer(nn.Module):
    """Simple linear layer followed by tanh."""
    def __init__(self, input_dim: int, output_dim: int = 1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(x))


# ----------------------------------------------------------------------
# Hybrid auto‑encoder definition
# ----------------------------------------------------------------------
class HybridAutoencoder(nn.Module):
    """Auto‑encoder with a quantum latent layer."""
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        latent_dim: int = 32,
        hidden_dims: tuple[int, int] = (128, 64),
        dropout: float = 0.1,
        quantum_reps: int = 3,
    ) -> None:
        super().__init__()
        self.input_shape = input_shape

        # Encoder
        self.encoder = nn.Sequential(
            ConvFilter(kernel_size=2, threshold=0.0),
            nn.Flatten(),
            nn.Linear((input_shape[1]-1)*(input_shape[2]-1), hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], latent_dim),
        )

        # Quantum latent processor
        self.quantum = HybridQNN(latent_dim=latent_dim, reps=quantum_reps)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], (input_shape[1]-1)*(input_shape[2]-1)),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = self.encoder(x)
        quantum_out = self.quantum(batch)
        recon = self.decoder(quantum_out)
        recon = recon.view(x.shape)
        return recon


# ----------------------------------------------------------------------
# Factory helper
# ----------------------------------------------------------------------
def HybridAutoencoderFactory(
    input_shape: tuple[int, int, int],
    *,
    latent_dim: int = 32,
    hidden_dims: tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    quantum_reps: int = 3,
) -> HybridAutoencoder:
    """Return a configured hybrid auto‑encoder."""
    return HybridAutoencoder(
        input_shape=input_shape,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        quantum_reps=quantum_reps,
    )


# ----------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------
def train_hybrid_autoencoder(
    model: HybridAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Train the hybrid auto‑encoder and return the loss history."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
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


# ----------------------------------------------------------------------
# Utility helper
# ----------------------------------------------------------------------
def _as_tensor(data: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data
    return torch.tensor(data, dtype=torch.float32)


__all__ = [
    "HybridAutoencoder",
    "HybridAutoencoderFactory",
    "train_hybrid_autoencoder",
]
