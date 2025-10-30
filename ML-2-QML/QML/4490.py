"""Hybrid quantum autoencoder.

The quantum module uses torchquantum to encode classical data into a quantum
state, processes it with a random layer, measures the latent vector, and then
decodes it with a classical linear head.  It reuses the regression dataset
generator from `QuantumRegression.py` and provides a training routine
compatible with the classical one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from torch.utils.data import DataLoader, TensorDataset, Dataset


def generate_superposition_data(num_wires: int, samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample states of the form cos(theta)|0...0> + e^{i phi} sin(theta)|1...1>."""
    thetas = 2 * torch.pi * torch.rand(samples)
    phis = 2 * torch.pi * torch.rand(samples)
    omega_0 = torch.zeros(2 ** num_wires, dtype=torch.cfloat)
    omega_0[0] = 1.0
    omega_1 = torch.zeros(2 ** num_wires, dtype=torch.cfloat)
    omega_1[-1] = 1.0
    states = torch.cos(thetas)[:, None] * omega_0 + torch.exp(1j * phis)[:, None] * torch.sin(thetas)[:, None] * omega_1
    labels = torch.sin(2 * thetas) * torch.cos(phis)
    return states, labels


class RegressionDataset(Dataset):
    """Quantum regression dataset using superposition states."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {"states": self.states[index], "target": self.labels[index]}


@dataclass
class AutoencoderConfig:
    """Configuration for the quantum autoencoder."""
    input_dim: int
    latent_dim: int = 32
    num_wires: int | None = None
    dropout: float = 0.1


class AutoencoderHybrid(tq.QuantumModule):
    """Quantum encoder + classical decoder hybrid autoencoder."""
    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.n_wires = config.num_wires or config.latent_dim

        # Quantum encoder: a simple Ry‑based circuit
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{self.n_wires}xRy"]
        )
        # Quantum feature extractor
        self.q_layer = tq.RandomLayer(n_ops=30, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Classical decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.n_wires, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, config.input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode `x` quantumly, then decode classically."""
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)
        self.encoder(qdev, x)
        self.q_layer(qdev)
        latent = self.measure(qdev)
        return self.decoder(latent)


def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    num_wires: int | None = None,
    dropout: float = 0.1,
) -> AutoencoderHybrid:
    """Factory mirroring the classical helper."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        num_wires=num_wires,
        dropout=dropout,
    )
    return AutoencoderHybrid(config)


def train_autoencoder(
    model: AutoencoderHybrid,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Training loop for the quantum autoencoder."""
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


def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


__all__ = [
    "AutoencoderHybrid",
    "Autoencoder",
    "train_autoencoder",
    "RegressionDataset",
    "generate_superposition_data",
]
