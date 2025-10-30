"""Hybrid quantum‑classical auto‑encoder using Pennylane."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List

import pennylane as qml
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _as_tensor(data: torch.Tensor | list[float]) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 4
    hidden_dims: Tuple[int,...] = (64, 32)
    dropout: float = 0.1
    num_qubits: int = 4
    q_layers: int = 2


class Autoencoder(nn.Module):
    """Hybrid auto‑encoder: classical encoder → quantum latent → classical decoder."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config

        # Classical encoder
        enc_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            enc_layers.append(nn.Linear(in_dim, hidden))
            enc_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                enc_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        enc_layers.append(nn.Linear(in_dim, config.num_qubits))
        self.encoder = nn.Sequential(*enc_layers)

        # Classical decoder
        dec_layers = []
        in_dim = config.num_qubits
        for hidden in reversed(config.hidden_dims):
            dec_layers.append(nn.Linear(in_dim, hidden))
            dec_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                dec_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        dec_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

        # Quantum circuit
        self.q_device = qml.device("default.qubit", wires=config.num_qubits)
        # Variational parameters
        self.q_weights = nn.Parameter(
            torch.randn(config.q_layers, config.num_qubits, 3, requires_grad=True)
        )

        @qml.qnode(self.q_device, interface="torch")
        def _qnode(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            qml.AngleEmbedding(x, wires=range(config.num_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(config.num_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(config.num_qubits)]

        self.qnode = _qnode

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Classical encoder + quantum latent state."""
        x = self.encoder(inputs)
        return self.qnode(x, self.q_weights)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Classical decoder."""
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        latent = self.encode(inputs)
        return self.decode(latent)


def train_qautoencoder(
    model: Autoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> List[float]:
    """Train the hybrid auto‑encoder using Torch autograd."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

    for epoch in range(epochs):
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


__all__ = ["Autoencoder", "AutoencoderConfig", "train_qautoencoder"]
