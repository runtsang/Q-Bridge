"""Quantum hybrid autoencoding regression model.

This module implements a quantum regression network that mirrors the classical
HybridRegressionAutoencoder.  The quantum encoder maps classical features to a
superposition state, a variational auto‑encoding circuit learns a compact
representation, and a classical head produces the regression output.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

# Dataset (identical to the classical version)
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

# Variational auto‑encoder layer
class QLatentLayer(tq.QuantumModule):
    """Parameterized circuit that learns a compact latent representation."""
    def __init__(self, num_wires: int, latent_dim: int):
        super().__init__()
        self.n_wires = num_wires
        self.latent_dim = latent_dim
        self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(num_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)

    def forward(self, qdev: tq.QuantumDevice) -> None:
        self.random_layer(qdev)
        for w in range(self.n_wires):
            self.rx(qdev, wires=w)
            self.ry(qdev, wires=w)

# Main model
class HybridRegressionAutoencoder(tq.QuantumModule):
    def __init__(self, num_wires: int, latent_dim: int = 4):
        super().__init__()
        self.n_wires = num_wires
        self.latent_dim = latent_dim
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.latent_layer = QLatentLayer(num_wires, latent_dim)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(latent_dim, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.latent_layer(qdev)
        features = self.measure(qdev)  # shape (bsz, n_wires)
        latent_features = features[:, : self.latent_dim]
        return self.head(latent_features).squeeze(-1)

# Training helper
def train_qmodel(
    model: HybridRegressionAutoencoder,
    dataset: RegressionDataset,
    epochs: int = 50,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> list[float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(
        list(model.parameters()), lr=lr, weight_decay=1e-5
    )
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for batch in loader:
            states = batch["states"].to(device)
            target = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(states)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * states.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

__all__ = [
    "HybridRegressionAutoencoder",
    "RegressionDataset",
    "QLatentLayer",
    "train_qmodel",
]
