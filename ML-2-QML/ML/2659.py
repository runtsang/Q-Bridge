"""Hybrid Autoencoder combining classical QCNN and quantum encoder."""

from __future__ import annotations

import torch
from torch import nn
from typing import Iterable, Tuple
from.QCNN import QCNNModel
from.quantum_autoencoder import HybridAutoencoder as QuantumHybridAutoencoder

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


class HybridAutoencoder(nn.Module):
    """Hybrid autoencoder with classical QCNN encoder and quantum latent representation."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 8,
        decoder_hidden_dims: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        # Classical QCNN encoder
        self.qcnn = QCNNModel()
        # Quantum encoder
        self.quantum_encoder = QuantumHybridAutoencoder(latent_dim=latent_dim)
        # Decoder: fully connected network
        decoder_layers = []
        in_dim = latent_dim
        for hidden in decoder_hidden_dims:
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                decoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Classical feature extraction
        x = self.qcnn(inputs)
        # Quantum latent representation
        latent = self.quantum_encoder(x)
        # Reconstruction
        return self.decoder(latent)


def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 8,
    decoder_hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> HybridAutoencoder:
    """Factory returning a configured hybrid autoencoder."""
    return HybridAutoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        decoder_hidden_dims=decoder_hidden_dims,
        dropout=dropout,
    )


def train_autoencoder(
    model: HybridAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Training loop for hybrid autoencoder."""
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


__all__ = ["Autoencoder", "train_autoencoder", "HybridAutoencoder"]
