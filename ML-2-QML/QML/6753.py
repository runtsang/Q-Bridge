"""Hybrid variational autoencoder using Pennylane."""

from __future__ import annotations

import pennylane as qml
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from typing import Iterable

def _as_tensor(data: torch.Tensor | Iterable[float]) -> torch.Tensor:
    if isinstance(data, torch.Tensor):
        return data
    return torch.as_tensor(data, dtype=torch.float32)

class Autoencoder__gen612(nn.Module):
    """Hybrid encoder: quantum feature map + variational circuit, classical decoder."""
    def __init__(self, input_dim: int, latent_dim: int, dev_name: str = "default.qubit"):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.dev = qml.device(dev_name, wires=latent_dim, shots=None)
        # Variational parameters
        self.params = nn.Parameter(torch.randn(latent_dim * 3, dtype=torch.float64))
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )
        self.qnode = qml.QNode(self._quantum_circuit, self.dev, interface="torch", diff_method="parameter-shift")

    def _feature_map(self, x):
        for i in range(self.input_dim):
            qml.RY(x[i], wires=i % self.latent_dim)

    def _variational(self, params):
        qml.layer(qml.RealAmplitudes, wires=range(self.latent_dim), reps=3, activation=None, params=params)

    def _quantum_circuit(self, x, params):
        self._feature_map(x)
        self._variational(params)
        return [qml.expval(qml.PauliZ(i)) for i in range(self.latent_dim)]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.qnode(x, self.params)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

def train_autoencoder_qml(
    model: Autoencoder__gen612,
    data: torch.Tensor,
    *,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
    val_split: float = 0.1,
    patience: int = 10,
) -> list[float]:
    """Train hybrid quantum‑classical autoencoder with early‑stopping."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    n = len(dataset)
    val_n = int(n * val_split)
    train_n = n - val_n
    train_ds, val_ds = random_split(dataset, [train_n, val_n])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []
    best_val = float("inf")
    epochs_no_improve = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= train_n
        history.append(epoch_loss)
        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                recon = model(batch)
                loss = loss_fn(recon, batch)
                val_loss += loss.item() * batch.size(0)
        val_loss /= val_n
        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    return history

__all__ = [
    "Autoencoder__gen612",
    "train_autoencoder_qml",
]
