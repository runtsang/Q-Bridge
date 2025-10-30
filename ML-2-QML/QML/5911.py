"""
Quantum‑enhanced hybrid autoencoder.

The module keeps the same public API as the classical version but replaces
the fixed quantum circuit with a trainable variational circuit implemented
with Pennylane.  The circuit is differentiable with respect to its
parameters, allowing end‑to‑end gradient optimisation together with the
classical encoder/decoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Iterable, Tuple, List

import pennylane as qml

# --------------------------------------------------------------------------- #
#   Classical autoencoder backbone (identical to the ML module)
# --------------------------------------------------------------------------- #

@dataclass
class HybridAutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class ClassicalAutoencoderNet(nn.Module):
    def __init__(self, cfg: HybridAutoencoderConfig) -> None:
        super().__init__()
        encoder_layers: List[nn.Module] = []
        in_dim = cfg.input_dim
        for hidden in cfg.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                encoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers: List[nn.Module] = []
        in_dim = cfg.latent_dim
        for hidden in reversed(cfg.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                decoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

# --------------------------------------------------------------------------- #
#   Trainable quantum latent layer
# --------------------------------------------------------------------------- #

class QuantumLatent(nn.Module):
    """Variational circuit that maps a latent vector into a statevector.
    Parameters are optimised with the parameter‑shift rule via Pennylane.
    """
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        # Trainable offsets that are added to the encoded angles
        self.params = nn.Parameter(torch.zeros(latent_dim))
        # Quantum device
        self.dev = qml.device("default.qubit", wires=latent_dim)

        @qml.qnode(self.dev, interface="torch")
        def circuit(z: torch.Tensor) -> torch.Tensor:
            for i in range(self.latent_dim):
                # Encode the classical latent vector and add a trainable offset
                qml.RX(z[i] + self.params[i], wires=i)
            # Simple entangling layer
            qml.layer(qml.templates.BasicEntanglerLayers, reps=2)
            # Return expectation values of Pauli‑Z
            return [qml.expval(qml.PauliZ(i)) for i in range(self.latent_dim)]

        self.circuit = circuit

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return a batch of expectation‑value vectors."""
        batch_out = []
        for i in range(z.shape[0]):
            batch_out.append(self.circuit(z[i]))
        return torch.stack(batch_out, dim=0)

# --------------------------------------------------------------------------- #
#   Hybrid autoencoder
# --------------------------------------------------------------------------- #

class HybridAutoEncoder(nn.Module):
    """Hybrid autoencoder with a trainable quantum latent layer."""
    def __init__(self, cfg: HybridAutoencoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.classical = ClassicalAutoencoderNet(cfg)
        self.quantum = QuantumLatent(cfg.latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.classical.encode(x)
        qz = self.quantum(z)
        recon = self.classical.decode(qz)
        return recon

# --------------------------------------------------------------------------- #
#   Factory & training utilities
# --------------------------------------------------------------------------- #

def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> HybridAutoEncoder:
    cfg = HybridAutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return HybridAutoEncoder(cfg)

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data
    return torch.as_tensor(data, dtype=torch.float32)

def train_hybrid_autoencoder(
    model: HybridAutoEncoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> List[float]:
    """Training loop that optimises both classical and quantum parameters."""
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
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

__all__ = [
    "Autoencoder",
    "HybridAutoencoderConfig",
    "HybridAutoEncoder",
    "train_hybrid_autoencoder",
]
