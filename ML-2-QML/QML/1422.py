"""Hybrid quantum-classical autoencoder using Pennylane."""

import pennylane as qml
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Iterable, Tuple


def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


class AutoencoderHybrid(nn.Module):
    """Quantum encoder + classical decoder autoencoder."""
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 4,
        hidden_dims: Tuple[int,...] = (32,),
        num_layers: int = 2,
        device: str = "default.qubit",
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.device = device
        self.seed = seed

        # Quantum device
        self.q_dev = qml.device(device, wires=input_dim)
        # Trainable parameters for variational layers
        self.params = nn.Parameter(torch.randn(num_layers, input_dim, 3))
        # Classical decoder
        decoder_layers = []
        in_dim = latent_dim
        for h in hidden_dims:
            decoder_layers.append(nn.Linear(in_dim, h))
            decoder_layers.append(nn.ReLU())
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # Quantum node
        self.qnode = qml.QNode(self._quantum_encoder, self.q_dev, interface="torch")

    def _quantum_encoder(self, x: torch.Tensor, *params) -> torch.Tensor:
        """Variational encoder that returns expectation values of Pauli‑Z."""
        # Encode data as rotations
        for i in range(self.input_dim):
            qml.RX(x[i], wires=i)
        # Apply variational layers
        for layer_idx, layer_params in enumerate(params):
            for i in range(self.input_dim):
                qml.Rot(*layer_params[i], wires=i)
            for i in range(self.input_dim - 1):
                qml.CNOT(wires=[i, i + 1])
        # Measure expectation values on first `latent_dim` wires
        return [qml.expval(qml.PauliZ(i)) for i in range(self.latent_dim)]

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return latent vector for each input in the batch."""
        # Split parameters per layer
        params_split = [self.params[i] for i in range(self.num_layers)]
        latents = torch.stack([self.qnode(x, *params_split) for x in inputs])
        return latents

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Reconstruct input from latent vector."""
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        latents = self.encode(inputs)
        return self.decode(latents)

    def loss(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Mean‑squared‑error reconstruction loss."""
        return nn.functional.mse_loss(recon, target, reduction="mean")


def train_autoencoder(
    model: AutoencoderHybrid,
    data: torch.Tensor,
    *,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
    early_stop_patience: int | None = None,
) -> list[float]:
    """Training loop for the hybrid autoencoder."""
    device = device or torch.device("cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: list[float] = []
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = model.loss(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)

        # Early stopping
        if early_stop_patience is not None:
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    break
    return history


__all__ = ["AutoencoderHybrid", "train_autoencoder"]
