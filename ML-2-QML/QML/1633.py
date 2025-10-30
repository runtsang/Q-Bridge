"""
Quantum‑enhanced autoencoder using PennyLane.

The quantum layer refines the latent representation via a variational circuit.
"""

from __future__ import annotations

import dataclasses
from typing import Iterable, Tuple

import pennylane as qml
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# --------------------------------------------------------------------------- #
# Utility
# --------------------------------------------------------------------------- #
def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current device."""
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
@dataclasses.dataclass
class AutoencoderConfig:
    """Configuration for :class:`Autoencoder__gen204`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    num_qubits: int = 4
    num_layers: int = 3


# --------------------------------------------------------------------------- #
# Quantum refinement via PennyLane
# --------------------------------------------------------------------------- #
class QuantumRefinement(nn.Module):
    """Variational circuit that refines a latent vector."""
    def __init__(self, num_qubits: int, num_layers: int, dev: qml.Device | None = None) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.dev = dev or qml.device("default.qubit", wires=num_qubits)

        # Trainable parameters
        self.weights = nn.Parameter(torch.randn(num_layers, num_qubits, 3))

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(x, weights):
            for i in range(num_qubits):
                qml.RX(x[i], wires=i)
            for layer in range(num_layers):
                for q in range(num_qubits):
                    qml.Rot(*weights[layer, q], wires=q)
                for q in range(num_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

        self.circuit = circuit

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        # Support batched latent vectors
        if latent.ndim == 1:
            latent = latent.unsqueeze(0)
        refined = []
        for i in range(latent.shape[0]):
            refined.append(self.circuit(latent[i], self.weights))
        return torch.stack(refined, dim=0)


# --------------------------------------------------------------------------- #
# Core model
# --------------------------------------------------------------------------- #
class Autoencoder__gen204(nn.Module):
    """Quantum‑enhanced autoencoder."""
    def __init__(self, config: AutoencoderConfig) -> None:
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

        # Quantum refinement
        self.quantum = QuantumRefinement(
            num_qubits=config.num_qubits,
            num_layers=config.num_layers,
        )

        # Decoder
        decoder_layers = []
        in_dim = config.num_qubits  # output of quantum circuit
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def refine_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return self.quantum(latent)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        latent = self.encode(inputs)
        refined = self.refine_latent(latent)
        return self.decode(refined)


# --------------------------------------------------------------------------- #
# Training helper
# --------------------------------------------------------------------------- #
def train_autoencoder(
    model: Autoencoder__gen204,
    data: torch.Tensor,
    *,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Train the quantum‑enhanced autoencoder."""
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
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)

    return history


__all__ = ["Autoencoder__gen204", "AutoencoderConfig", "train_autoencoder"]
