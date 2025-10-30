"""Hybrid quantum-classical autoencoder using PennyLane."""

import pennylane as qml
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Iterable, Tuple, List, Optional

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


@dataclass
class AutoencoderConfig:
    """Configuration for the hybrid autoencoder."""
    input_dim: int
    latent_dim: int = 8
    hidden_dims: Tuple[int,...] = (64, 32)
    num_wires: int = 4
    reps: int = 3
    dropout: float = 0.1
    device: torch.device | None = None


class Autoencoder(nn.Module):
    """Hybrid quantum-classical autoencoder."""
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Quantum device and circuit
        self.q_device = qml.device("default.qubit", wires=cfg.num_wires)
        self.qnode = self._create_qnode()

        # Classical decoder
        decoder_layers = []
        in_dim = cfg.latent_dim
        for hidden in cfg.hidden_dims:
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                decoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        self.to(self.device)

    def _create_qnode(self) -> qml.QNode:
        @qml.qnode(self.q_device, interface="torch")
        def circuit(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            # Feature map
            qml.AngleEmbedding(x, wires=range(self.cfg.num_wires))
            # Variational ansatz
            qml.StronglyEntanglingLayers(weights, wires=range(self.cfg.num_wires))
            # Return expectation values of PauliZ on each wire
            return [qml.expval(qml.PauliZ(i)) for i in range(self.cfg.num_wires)]
        return circuit

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input into a quantum latent vector."""
        if not hasattr(self, "_weights"):
            weight_shapes = qml.StronglyEntanglingLayers.shape(self.cfg.reps, self.cfg.num_wires)
            self._weights = nn.Parameter(torch.randn(weight_shapes))
        latent = self.qnode(x, self._weights)
        return latent

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        latent = self.encode(x)
        return self.decode(latent)


def train_autoencoder(
    model: Autoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> List[float]:
    """Training loop for the hybrid autoencoder."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

    loader = DataLoader(TensorDataset(_as_tensor(data)), batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch, in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(loader.dataset)
        history.append(epoch_loss)
    return history


__all__ = ["Autoencoder", "AutoencoderConfig", "train_autoencoder"]
