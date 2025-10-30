"""Quantum hybrid autoencoder combining classical encoder‑decoder with a variational quantum circuit.

The architecture merges ideas from:
- A fully‑connected autoencoder (PyTorch) for feature extraction.
- A quantum encoder that maps classical latent vectors to quantum expectation values.
- A quantum kernel that can be used for regularisation or downstream similarity queries.

The module exposes a factory :func:`QuantumHybridAutoencoder` that returns a ready‑to‑train
:class:`QuantumHybridAutoencoderNet`.  The training loop is adapted from the original
``train_autoencoder`` but includes a call to the quantum encoder during forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Import quantum primitives from the companion QML module
from.quantum_autoencoder_qml import QuantumEncoder, QuantumKernel


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
class QuantumHybridAutoencoderConfig:
    """Configuration values for :class:`QuantumHybridAutoencoderNet`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    n_qubits: int | None = None  # defaults to latent_dim if None


class QuantumHybridAutoencoderNet(nn.Module):
    """A hybrid classical‑quantum autoencoder.

    The classical encoder produces a latent vector that is then fed into a
    variational quantum circuit.  The circuit outputs a new latent vector
    (expectation values of Pauli‑Z) which is decoded back to the input space.
    """
    def __init__(self, config: QuantumHybridAutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        hidden_dims = config.hidden_dims
        in_dim = config.input_dim

        # Classical encoder
        encoder_layers = []
        for hidden in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Quantum encoder
        n_qubits = config.n_qubits or config.latent_dim
        self.quantum_encoder = QuantumEncoder(n_qubits)

        # Classical decoder
        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Classical encoder only."""
        return self.encoder(inputs)

    def quantum_encode(self, latent: torch.Tensor) -> torch.Tensor:
        """Pass classical latent vector through the quantum circuit."""
        # The quantum circuit expects a NumPy array; convert and keep device
        latent_np = latent.detach().cpu().numpy()
        q_latent_np = self.quantum_encoder(latent_np)
        return torch.tensor(q_latent_np, dtype=torch.float32, device=latent.device)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        latent = self.encode(inputs)
        q_latent = self.quantum_encode(latent)
        return self.decode(q_latent)


def QuantumHybridAutoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    n_qubits: int | None = None,
) -> QuantumHybridAutoencoderNet:
    """Factory that mirrors the original ``Autoencoder`` but creates a hybrid version."""
    config = QuantumHybridAutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        n_qubits=n_qubits,
    )
    return QuantumHybridAutoencoderNet(config)


def train_autoencoder_qml(
    model: QuantumHybridAutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
    verbose: bool = False,
) -> list[float]:
    """Simple reconstruction training loop for the hybrid autoencoder."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for epoch in range(epochs):
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
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} – loss: {epoch_loss:.6f}")
    return history


__all__ = [
    "QuantumHybridAutoencoder",
    "QuantumHybridAutoencoderConfig",
    "QuantumHybridAutoencoderNet",
    "train_autoencoder_qml",
]
