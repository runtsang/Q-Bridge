"""Hybrid quantum autoencoder with a variational encoder and classical decoder."""
from __future__ import annotations

import torch
from torch import nn
import torchquantum as tq
import torchquantum.functional as tqf
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from dataclasses import dataclass
from typing import Iterable, Tuple

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Convert input to a float32 tensor on the default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate complex superposition states and sinusoidal targets."""
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
    return states, labels

class RegressionDataset(torch.utils.data.Dataset):
    """Dataset wrapping the complex superposition data."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

@dataclass
class HybridAutoencoderConfig:
    """Configuration for the quantum hybrid autoencoder."""
    num_wires: int
    latent_dim: int = 32
    hidden_dims: Tuple[int,...] = (128, 64)
    dropout: float = 0.1

class HybridAutoencoder(tq.QuantumModule):
    """Quantum‑classical autoencoder: variational encoder + classical MLP decoder."""
    class QEncoder(tq.QuantumModule):
        def __init__(self, num_wires: int, latent_dim: int):
            super().__init__()
            self.n_wires = num_wires
            self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.measure = tq.MeasureAll(tq.PauliZ)
            self.latent_linear = nn.Linear(num_wires, latent_dim)

        def forward(self, qdev: tq.QuantumDevice):
            self.encoder(qdev, qdev.state)
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)
            raw = self.measure(qdev)
            raw = raw.to(torch.float32)
            return self.latent_linear(raw)

    def __init__(self, config: HybridAutoencoderConfig):
        super().__init__()
        self.config = config
        self.encoder = self.QEncoder(config.num_wires, config.latent_dim)

        # Classical MLP decoder matching the latent dimension
        mlp_layers = []
        in_dim = config.latent_dim
        for hidden in config.hidden_dims:
            mlp_layers.append(nn.Linear(in_dim, hidden))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        mlp_layers.append(nn.Linear(in_dim, config.num_wires))
        self.decoder = nn.Sequential(*mlp_layers)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.config.num_wires, bsz=bsz, device=state_batch.device)
        latent = self.encoder(qdev)
        reconstructed = self.decoder(latent)
        return reconstructed

def train_hybrid_autoencoder_quantum(
    model: HybridAutoencoder,
    dataset: torch.utils.data.Dataset,
    *,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Training loop for the quantum hybrid autoencoder."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for batch in loader:
            states = batch["states"].to(device)
            target = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(states)
            loss = loss_fn(recon, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * states.size(0)
        epoch_loss /= len(loader.dataset)
        history.append(epoch_loss)
    return history

__all__ = [
    "HybridAutoencoder",
    "HybridAutoencoderConfig",
    "train_hybrid_autoencoder_quantum",
    "RegressionDataset",
    "generate_superposition_data",
]
