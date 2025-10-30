"""Hybrid quantum regression model that fuses a classical encoder, a variational quantum layer, and a classical readout."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from typing import Tuple, List

def generate_superposition_data(num_wires: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data in a superposition basis with a sinusoidal label."""
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

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {"states": torch.tensor(self.states[idx], dtype=torch.cfloat),
                "target": torch.tensor(self.labels[idx], dtype=torch.float32)}

class HybridRegression(tq.QuantumModule):
    """Quantum‑classical hybrid regression network."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random = tq.RandomLayer(n_ops=40, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random(qdev)
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)
                self.rz(qdev, wires=w)

    def __init__(self, num_wires: int, latent_dim: int = 32):
        super().__init__()
        self.n_wires = num_wires
        # Classical encoder (mirrors the classical encoder)
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        # Measurement resembling EstimatorQNN: PauliZ expectation
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Classical readout head
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

def train_quantum(
    model: tq.QuantumModule,
    dataloader: torch.utils.data.DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    device: torch.device = None
) -> List[float]:
    """Training loop for the hybrid quantum regression model."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    history: List[float] = []

    for epoch in range(epochs):
        total = 0.0
        model.train()
        for batch in dataloader:
            xs = batch["states"].to(device)
            ys = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            preds = model(xs)
            loss = loss_fn(preds, ys)
            loss.backward()
            optimizer.step()
            total += loss.item() * xs.size(0)
        epoch_loss = total / len(dataloader.dataset)
        history.append(epoch_loss)
    return history

__all__ = ["HybridRegression", "RegressionDataset", "generate_superposition_data", "train_quantum"]
