"""
Autoencoder__gen326.py – Quantum VAE using PennyLane.

This implementation mirrors the classical VAE but replaces the encoder/decoder
with a parameterised quantum circuit.  The latent distribution is encoded
into a set of qubits via a RealAmplitudes ansatz and decoded by a second
ansatz.  Training uses stochastic gradient descent with the Adam optimiser
and a hybrid loss that combines a reconstruction term (cross‑entropy) with
a KL‑divergence penalty on the latent qubit amplitudes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List

import pennylane as qml
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #
def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class AutoencoderConfig:
    """Hyper‑parameters for the quantum VAE."""
    input_dim: int
    latent_dim: int = 3
    hidden_layers: int = 2
    reps: int = 3
    dropout: float = 0.0
    latent_loss_weight: float = 1e-3


# --------------------------------------------------------------------------- #
# Core network
# --------------------------------------------------------------------------- #
class AutoencoderNet(nn.Module):
    """
    Quantum Variational Autoencoder.

    The encoder maps classical inputs to a latent qubit register using a
    RealAmplitudes ansatz.  The decoder reconstructs a probability
    distribution over the input space.  The network is trained with an
    Adam optimiser and a hybrid loss.
    """
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Quantum device
        self.dev = qml.device("default.qubit", wires=cfg.latent_dim + cfg.input_dim)

        # Encoder ansatz
        def encoder_circuit(x: np.ndarray):
            qml.templates.BasicEntanglerLayers(weights=x, wires=range(cfg.latent_dim))

        # Decoder ansatz
        def decoder_circuit(z: np.ndarray):
            qml.templates.RealAmplitudes(weights=z, wires=range(cfg.latent_dim, cfg.latent_dim + cfg.input_dim))

        self.encoder = qml.QNode(encoder_circuit, self.dev, interface="torch")
        self.decoder = qml.QNode(decoder_circuit, self.dev, interface="torch")

    # --------------------------------------------------------------------- #
    # Forward pass
    # --------------------------------------------------------------------- #
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return latent amplitudes (probabilities) as a torch tensor."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Return reconstruction probabilities."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (reconstruction, latent)."""
        latent = self.encode(x)
        recon = self.decode(latent)
        return recon, latent


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 3,
    hidden_layers: int = 2,
    reps: int = 3,
    dropout: float = 0.0,
    latent_loss_weight: float = 1e-3,
) -> AutoencoderNet:
    cfg = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_layers=hidden_layers,
        reps=reps,
        dropout=dropout,
        latent_loss_weight=latent_loss_weight,
    )
    return AutoencoderNet(cfg)


# --------------------------------------------------------------------------- #
# Training routine
# --------------------------------------------------------------------------- #
def train_autoencoder(
    model: AutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
    early_stop_patience: int | None = None,
) -> List[Tuple[float, float]]:
    """
    Train the quantum VAE.

    Returns a history of (recon_loss, kl_loss) for each epoch.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    recon_loss_fn = nn.BCELoss(reduction="sum")
    history: List[Tuple[float, float]] = []

    best_recon = float("inf")
    patience = 0

    for epoch in range(epochs):
        epoch_recon, epoch_kl = 0.0, 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            recon, latent = model(batch)

            recon_loss = recon_loss_fn(recon, batch)
            # KL divergence between latent distribution and uniform prior
            kl_loss = torch.sum(latent * torch.log(latent + 1e-10))
            loss = recon_loss + model.cfg.latent_loss_weight * kl_loss

            loss.backward()
            optimizer.step()

            epoch_recon += recon_loss.item()
            epoch_kl += kl_loss.item()

        epoch_recon /= len(dataset)
        epoch_kl /= len(dataset)
        history.append((epoch_recon, epoch_kl))

        # Early stopping on reconstruction loss
        if early_stop_patience is not None:
            if epoch_recon < best_recon:
                best_recon = epoch_recon
                patience = 0
            else:
                patience += 1
                if patience >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

    return history


__all__ = ["Autoencoder", "AutoencoderConfig", "AutoencoderNet", "train_autoencoder"]
