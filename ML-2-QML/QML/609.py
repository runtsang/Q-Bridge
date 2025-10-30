"""Quantum regression dataset and model with advanced training utilities and optional error mitigation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchquantum as tq

def generate_superposition_data(num_wires: int, samples: int, noise_std: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Sample states |ψ⟩ = cosθ|0…0⟩ + e^{iϕ} sinθ|1…1⟩ with optional label noise."""
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
    if noise_std > 0.0:
        labels += np.random.normal(0.0, noise_std, size=labels.shape)
    return states, labels.astype(np.float32)

class RegressionDataset(Dataset):
    def __init__(self, samples: int, num_wires: int, noise_std: float = 0.0):
        self.states, self.labels = generate_superposition_data(num_wires, samples, noise_std)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class RegressionModel(tq.QuantumModule):
    """
    Variational quantum regression network with a random layer and per‑wire RX/RY rotations.
    Includes a simple training loop with early stopping and LR scheduler.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int, n_ops: int = 30):
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, num_wires: int, encoder_name: str | None = None):
        super().__init__()
        self.n_wires = num_wires
        if encoder_name is None:
            encoder_name = f"{num_wires}xRy"
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[encoder_name])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 200,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        patience: int = 10,
        device: str | torch.device = "cpu",
    ) -> None:
        """Train with MSE loss, early stopping, and LR scheduler."""
        self.to(device)
        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=patience//2, verbose=False)
        criterion = nn.MSELoss()
        best_val = float("inf")
        best_state = None
        for epoch in range(epochs):
            self.train()
            for batch in train_loader:
                optimizer.zero_grad()
                pred = self(batch["states"].to(device))
                loss = criterion(pred, batch["target"].to(device))
                loss.backward()
                optimizer.step()
            if val_loader is not None:
                val_loss = self.evaluate(val_loader, device)
                scheduler.step(val_loss)
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                elif epoch - (best_val - val_loss) > patience:
                    break
        if best_state is not None:
            self.load_state_dict(best_state)

    def evaluate(self, loader: DataLoader, device: str | torch.device) -> float:
        self.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for batch in loader:
                pred = self(batch["states"].to(device))
                total += ((pred - batch["target"].to(device)) ** 2).sum().item()
                count += batch["target"].numel()
        return total / count

    def predict(self, X: torch.Tensor, device: str | torch.device = "cpu") -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self(X.to(device)).cpu()

__all__ = ["RegressionModel", "RegressionDataset", "generate_superposition_data"]
