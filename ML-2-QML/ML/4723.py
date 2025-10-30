"""Hybrid autoencoder integrating classical MLP and a fully‑connected quantum encoder.

The implementation stitches together ideas from the classic autoencoder,
the FCL quantum layer, and the fraud‑detection layered structure.
It uses PennyLane for the variational circuit and PyTorch for the decoder.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn
import pennylane as qml
from pennylane import numpy as pnp


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
class AutoencoderConfig:
    """Configuration values for the hybrid autoencoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    quantum_num_qubits: int | None = None  # if None, set to latent_dim
    quantum_reps: int = 3
    quantum_device: str = "default.qubit"


class QuantumEncoder(nn.Module):
    """Variational quantum encoder based on angle embedding + RealAmplitudes."""
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        num_qubits: int,
        reps: int = 3,
        device_name: str = "default.qubit",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_qubits = num_qubits
        self.reps = reps
        self.device = qml.device(device_name, wires=num_qubits)

        # Parameter vector for the variational part
        self.var_params = nn.Parameter(
            torch.randn(reps, num_qubits, 2) * 0.1, dtype=torch.float64
        )

        @qml.qnode(self.device, interface="torch", diff_method="backprop")
        def circuit(x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
            # Encode classical data via angle embedding
            qml.AngleEmbedding(x[: self.num_qubits], wires=range(self.num_qubits))
            # Variational layers
            for r in range(reps):
                qml.RealAmplitudes(params[r], wires=range(self.num_qubits))
            # Return expectation values of PauliZ for each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        self.circuit = circuit

    def forward(self, input_vec: torch.Tensor) -> torch.Tensor:
        """Encode input into a latent vector."""
        # Pad or truncate to match number of qubits
        vec = input_vec
        if vec.shape[-1] > self.num_qubits:
            vec = vec[..., : self.num_qubits]
        elif vec.shape[-1] < self.num_qubits:
            pad = torch.zeros(*vec.shape[:-1], self.num_qubits - vec.shape[-1], device=vec.device)
            vec = torch.cat([vec, pad], dim=-1)
        return self.circuit(vec, self.var_params)


class HybridAutoencoder(nn.Module):
    """Hybrid autoencoder with a quantum encoder and classical decoder."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config

        # Quantum encoder
        num_qubits = config.quantum_num_qubits or config.latent_dim
        self.quantum_encoder = QuantumEncoder(
            input_dim=config.input_dim,
            latent_dim=config.latent_dim,
            num_qubits=num_qubits,
            reps=config.quantum_reps,
            device_name=config.quantum_device,
        )

        # Classical decoder MLP
        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in config.hidden_dims:
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return latent representation of inputs."""
        return self.quantum_encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Reconstruct input from latent vector."""
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))


def train_autoencoder(
    model: HybridAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Training loop for the hybrid autoencoder."""
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


__all__ = ["AutoencoderConfig", "QuantumEncoder", "HybridAutoencoder", "train_autoencoder"]
