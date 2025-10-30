"""
AutoencoderV2: Variational autoencoder built with Pennylane.
Provides the same high‑level API as the classical counterpart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Optional

import pennylane as qml
import pennylane.numpy as np
from pennylane import numpy as pnp
from pennylane import qnn
import torch


@dataclass
class AutoencoderV2Config:
    """Configuration for the quantum autoencoder."""

    input_dim: int
    latent_dim: int = 3
    hidden_layers: Tuple[int,...] = (4, 4)
    num_qubits: int = 5
    device: str = "default.qubit"  # or "mq" for real hardware
    shots: int = 1024


class AutoencoderV2:
    """Quantum autoencoder using a Pennylane variational circuit."""

    def __init__(self, cfg: AutoencoderV2Config) -> None:
        self.cfg = cfg
        self.dev = qml.device(cfg.device, wires=cfg.num_qubits, shots=cfg.shots)

        # Build feature map (simple linear embedding)
        def feature_map(x):
            for i, val in enumerate(x):
                qml.RX(val, wires=i)
            return x

        # Variational ansatz
        def ansatz(params):
            for i in range(cfg.latent_dim):
                qml.RY(params[i], wires=i)
            for layer in range(len(cfg.hidden_layers)):
                for i in range(cfg.latent_dim):
                    qml.CNOT(wires=[i, (i + 1) % cfg.latent_dim])
                for i in range(cfg.latent_dim):
                    qml.RZ(params[cfg.latent_dim + layer * cfg.latent_dim + i], wires=i)

        # Combine into a QNode
        @qml.qnode(self.dev, interface="torch")
        def circuit(x, params):
            feature_map(x)
            ansatz(params)
            return [qml.expval(qml.PauliZ(i)) for i in range(cfg.latent_dim)]

        # Trainable parameters
        self.params = torch.nn.Parameter(
            torch.randn(cfg.latent_dim + cfg.latent_dim * len(cfg.hidden_layers))
        )

        self.circuit = circuit
        self.loss_fn = torch.nn.MSELoss()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return latent representation for a single input sample."""
        return self.circuit(x, self.params)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct the input from the latent vector.
        For simplicity we return a linear mapping using a classical layer.
        """
        # A very small classical decoder
        if not hasattr(self, "_decoder"):
            self._decoder = torch.nn.Linear(self.cfg.latent_dim, self.cfg.input_dim)
        return self._decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)

    def train_step(self, x_batch: torch.Tensor, optimizer: torch.optim.Optimizer) -> torch.Tensor:
        optimizer.zero_grad()
        recon = self.forward(x_batch)
        loss = self.loss_fn(recon, x_batch)
        loss.backward()
        optimizer.step()
        return loss


def train_autoencoder_v2_qml(
    model: AutoencoderV2,
    data: torch.Tensor,
    *,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 0.01,
    device: str | None = None,
) -> List[float]:
    """Train the quantum autoencoder using Pennylane and torch autograd."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history: List[float] = []

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(_as_tensor(data)), batch_size=batch_size, shuffle=True
    )

    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device or torch.device("cpu"))
            loss = model.train_step(batch, optimizer)
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(loader.dataset)
        loss_history.append(epoch_loss)
        print(f"Epoch {epoch+1}/{epochs} – loss: {epoch_loss:.4f}")

    return loss_history


__all__ = ["AutoencoderV2", "AutoencoderV2Config", "train_autoencoder_v2_qml"]
