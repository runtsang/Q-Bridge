"""Hybrid quantum‑classical autoencoder using PennyLane."""

import pennylane as qml
import torch
from torch import nn
import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class AutoencoderConfig:
    """Configuration for :class:`Autoencoder`."""
    input_dim: int
    latent_dim: int = 4
    hidden_dims: Tuple[int,...] = (8, 4)
    dropout: float = 0.0
    device: str = "default.qubit"
    wires: int | None = None


class Autoencoder(nn.Module):
    """Quantum encoder + classical decoder."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.wires = config.wires or config.latent_dim
        self.dev = qml.device(config.device, wires=self.wires)

        # Variational encoder
        self.encoder_params = nn.Parameter(
            torch.randn(1, self.wires, 2)  # reps=1, 2 parameters per layer
        )

        # Classical decoder
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

        # QNode
        @qml.qnode(self.dev, interface="torch")
        def _qnode(inputs: torch.Tensor, params: torch.Tensor):
            # Encode data into rotation angles
            for i, w in enumerate(self.wires):
                qml.RY(inputs[i], wires=w)
            # Variational layers
            for layer_params in params[0]:
                qml.RY(layer_params[0], wires=layer_params[1])
                qml.RZ(layer_params[1], wires=layer_params[1])
            return [qml.expval(qml.PauliZ(w)) for w in range(self.wires)]

        self._qnode = _qnode

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Map classical input to latent representation via quantum circuit."""
        # Pad or truncate to match input_dim
        if inputs.shape[-1]!= self.config.input_dim:
            raise ValueError("Input dimension mismatch.")
        return self._qnode(inputs, self.encoder_params)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(inputs))

    def train_step(
        self,
        data: torch.Tensor,
        *,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 32,
    ) -> list[float]:
        """Simple training loop for the hybrid autoencoder."""
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        history: list[float] = []

        dataset = torch.utils.data.TensorDataset(data)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for _ in range(epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                optimizer.zero_grad()
                recon = self(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.size(0)
            epoch_loss /= len(dataset)
            history.append(epoch_loss)
        return history


def AutoencoderFactory(
    input_dim: int,
    *,
    latent_dim: int = 4,
    hidden_dims: Tuple[int,...] = (8, 4),
    dropout: float = 0.0,
    device: str = "default.qubit",
    wires: int | None = None,
) -> Autoencoder:
    """Return a configured hybrid :class:`Autoencoder` instance."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        device=device,
        wires=wires,
    )
    return Autoencoder(config)


__all__ = ["Autoencoder", "AutoencoderFactory", "AutoencoderConfig"]
