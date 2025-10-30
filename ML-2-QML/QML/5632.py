"""Quantum regression with a variational autoencoder front‑end.

The quantum model first maps the classical feature vector onto a
set of qubits using a GeneralEncoder.  A variational autoencoder
compresses the full state into a smaller latent register.  The latent
amplitudes are then interpreted as classical features and fed into
a small quantum regression layer that outputs a single value via
Pauli‑Z measurement.  The architecture mirrors the classical
pipeline while preserving a fully quantum training loop.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Tuple

import torchquantum as tq


def generate_superposition_data(
    num_wires: int, samples: int, seed: int | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset of complex superposition states and a smooth
    target function.  The state vector is constructed as
    cos(theta)|0…0> + e^{i phi} sin(theta)|1…1>.
    """
    rng = np.random.default_rng(seed)
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * rng.random(samples)
    phis = 2 * np.pi * rng.random(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)


class RegressionDataset(Dataset):
    """
    Dataset that returns complex state vectors and a scalar target.
    """
    def __init__(self, samples: int, num_wires: int, seed: int | None = None):
        self.states, self.labels = generate_superposition_data(num_wires, samples, seed)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class RegressionAutoencoder(tq.QuantumModule):
    """
    Quantum regression model that incorporates a variational autoencoder
    front‑end.  The model is fully differentiable via torchquantum
    and can be trained with standard PyTorch optimizers.
    """

    class QAutoencoder(tq.QuantumModule):
        """
        Variational autoencoder that compresses a full register into
        a smaller latent register using a random layer followed by
        trainable single‑qubit rotations.
        """
        def __init__(self, total_wires: int, latent_wires: int):
            super().__init__()
            self.total_wires = total_wires
            self.latent_wires = latent_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(total_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
            self.random_layer(qdev)
            for w in range(self.total_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)
            features = self.measure(qdev)
            # keep only the first `latent_wires` qubits as the latent vector
            return features[:, :self.latent_wires]

    def __init__(self, total_wires: int, latent_wires: int):
        super().__init__()
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{total_wires}xRy"])
        self.autoencoder = self.QAutoencoder(total_wires, latent_wires)
        self.regressor = nn.Linear(latent_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        `state_batch` is a batch of complex state vectors.
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.autoencoder.total_wires, bsz=bsz, device=state_batch.device
        )
        # Encode the classical data into the quantum register
        self.encoder(qdev, state_batch)
        # Compress to latent representation
        latent = self.autoencoder(qdev)
        # Classical regression head
        return self.regressor(latent).squeeze(-1)


def train_qmodel(
    model: RegressionAutoencoder,
    dataset: Dataset,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """
    Standard training loop for the quantum regression model.
    The entire model, including the variational autoencoder, is
    updated jointly using back‑propagation through the quantum
    operations.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for batch in loader:
            states = batch["states"].to(device)
            targets = batch["target"].to(device)
            optimizer.zero_grad()
            preds = model(states)
            loss = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * states.size(0)
        epoch_loss /= len(loader.dataset)
        history.append(epoch_loss)
    return history


__all__ = [
    "RegressionAutoencoder",
    "RegressionDataset",
    "generate_superposition_data",
    "train_qmodel",
]
