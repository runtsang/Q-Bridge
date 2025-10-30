"""Hybrid classical‑quantum autoencoder with staged training and configurable quantum decoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pennylane as qml
import pennylane.numpy as npq

__all__ = [
    "AutoencoderConfig",
    "HybridAutoencoder",
    "train_hybrid_autoencoder",
    "pretrain_classical",
    "pretrain_quantum",
    "HybridAutoencoderFactory",
]


@dataclass
class AutoencoderConfig:
    """Configuration values for :class:`HybridAutoencoder`."""

    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

    # Quantum‑specific knobs
    quantum_decoder: str = "ry_rz"  # "ry_rz" or "real_amplitudes"
    qdevice: str = "default.qubit"  # Pennylane device name
    reps: int = 3  # number of variational layers
    seed: int = 42


class HybridAutoencoder(nn.Module):
    """Hybrid encoder–decoder with a classical encoder and a quantum decoder."""

    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        self.config = config

        # Classical encoder
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

        # Quantum decoder
        self.latent_dim = config.latent_dim
        dev = qml.device(config.qdevice, wires=self.latent_dim)

        def _qnode(latent: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
            # Angle embedding of the latent vector
            qml.templates.AngleEmbedding(latent, wires=range(self.latent_dim))
            # Variational layer
            if config.quantum_decoder == "ry_rz":
                # Use two sets of rotations per layer
                for r in range(config.reps):
                    qml.templates.AngleEmbedding(
                        params[r * self.latent_dim : (r + 1) * self.latent_dim],
                        wires=range(self.latent_dim),
                    )
                    qml.templates.AngleEmbedding(
                        params[(config.reps + r) * self.latent_dim : (config.reps + r + 1) * self.latent_dim],
                        wires=range(self.latent_dim),
                    )
            else:  # real_amplitudes
                qml.templates.RealAmplitudes(
                    params, wires=range(self.latent_dim), reps=config.reps
                )
            # Return expectation values of PauliZ on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.latent_dim)]

        self.qnode = qml.QNode(_qnode, dev, interface="torch")

        # Determine number of variational parameters
        if config.quantum_decoder == "ry_rz":
            num_params = config.reps * 2 * self.latent_dim
        else:
            num_params = config.reps * self.latent_dim * 2  # RealAmplitudes uses 2 parameters per qubit per layer
        self.qparams = nn.Parameter(torch.randn(num_params))

        # Classical decoder mapping quantum output back to input space
        self.decoder = nn.Linear(self.latent_dim, config.input_dim)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def quantum_decode(self, latent: torch.Tensor) -> torch.Tensor:
        # Process each sample in the batch individually
        batch_size = latent.shape[0]
        outputs = []
        for i in range(batch_size):
            out = self.qnode(latent[i], self.qparams)
            outputs.append(out)
        return torch.stack(outputs)

    def decode(self, q_outputs: torch.Tensor) -> torch.Tensor:
        return self.decoder(q_outputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        latent = self.encode(inputs)
        q_outputs = self.quantum_decode(latent)
        recon = self.decode(q_outputs)
        return recon


def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


def _train_model(
    model: nn.Module,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
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
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


def pretrain_classical(
    model: HybridAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    # Freeze quantum decoder
    for param in model.qparams:
        param.requires_grad = False
    return _train_model(
        model,
        data,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
    )


def pretrain_quantum(
    model: HybridAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    # Freeze classical encoder and decoder
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.decoder.parameters():
        param.requires_grad = False
    return _train_model(
        model,
        data,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
    )


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
    # All parameters trainable
    for param in model.parameters():
        param.requires_grad = True
    return _train_model(
        model,
        data,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
    )


def HybridAutoencoderFactory(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    quantum_decoder: str = "ry_rz",
    qdevice: str = "default.qubit",
    reps: int = 3,
    seed: int = 42,
) -> HybridAutoencoder:
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        quantum_decoder=quantum_decoder,
        qdevice=qdevice,
        reps=reps,
        seed=seed,
    )
    return HybridAutoencoder(config)
