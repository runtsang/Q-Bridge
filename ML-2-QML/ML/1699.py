import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Tuple

import pennylane as qml
from pennylane import numpy as np


@dataclass
class AutoencoderConfig:
    """Configuration for the hybrid variational autoencoder."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    num_q_nodes: int = 1  # number of quantum nodes used in the encoder/decoder
    device: torch.device | None = None


class AutoencoderNet(nn.Module):
    """Hybrid ML‑classical autoencoder with a quantum encoder‑decoder."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
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

        # Quantum encoder
        self.qnode = qml.QNode(
            self._quantum_encode,
            device="default.qubit",
            interface="torch",
        )
        # Classical decoder
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

    def _quantum_encode(self, latent: torch.Tensor, weights: torch.Tensor):
        """Quantum circuit that refines the latent vector."""
        dev = qml.device("default.qubit", wires=latent.shape[-1] + 1)
        # Encode latent into first n qubits
        for i, val in enumerate(latent):
            qml.PhaseShift(val, wires=i)
        # Apply a parameterized entangling layer
        for i in range(latent.shape[-1] - 1):
            qml.CNOT(wires=[i, i + 1])
        # Parameterized rotation on an aux qubit
        qml.RX(weights[0], wires=latent.shape[-1])
        return qml.expval(qml.PauliZ(latent.shape[-1]))

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Classical + quantum encoding."""
        latent = self.encoder(inputs)
        # Random weights for the quantum layer (trained as part of the network)
        weights = torch.randn(1, device=latent.device)
        return self.qnode(latent, weights)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.decode(self.encode(inputs))

def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    num_q_nodes: int = 1,
    device: torch.device | None = None,
) -> AutoencoderNet:
    """Factory that creates a hybrid autoencoder."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        num_q_nodes=num_q_nodes,
        device=device,
    )
    return AutoencoderNet(config)

def train_autoencoder(
    model: AutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Training loop that optimizes both classical and quantum parameters."""
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

__all__ = ["Autoencoder", "AutoencoderConfig", "AutoencoderNet", "train_autoencoder"]
