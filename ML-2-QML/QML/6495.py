"""Quantum regression model with a variational circuit, noise simulation, and dropout."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

def generate_superposition_data(num_wires: int, samples: int, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
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

class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, samples: int, num_wires: int, seed: int | None = None):
        self.states, self.labels = generate_superposition_data(num_wires, samples, seed)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QuantumRegression__gen324(tq.QuantumModule):
    """Quantum regression with a parameterized circuit, dropout, and optional noise."""

    class VariationalLayer(tq.QuantumModule):
        def __init__(self, num_wires: int, layers: int = 3):
            super().__init__()
            self.num_wires = num_wires
            self.layers = layers
            self.cnot = tq.CNOT
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            for _ in range(self.layers):
                # Entangling layer
                for wire in range(self.num_wires - 1):
                    self.cnot(qdev, wires=[wire, wire + 1])
                # Rotation layer
                for wire in range(self.num_wires):
                    self.rx(qdev, wires=wire)
                    self.ry(qdev, wires=wire)
                    self.rz(qdev, wires=wire)

    def __init__(self, num_wires: int = 4, dropout_p: float = 0.1):
        super().__init__()
        self.num_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.var_layer = self.VariationalLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.dropout = nn.Dropout(dropout_p)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.num_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.var_layer(qdev)
        features = self.measure(qdev)
        features = self.dropout(features)
        return self.head(features).squeeze(-1)

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader | None = None,
        epochs: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        patience: int = 10,
    ) -> "QuantumRegression__gen324":
        """Simple training loop for the quantum model."""
        device = next(self.parameters()).device
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience // 2, factor=0.5, verbose=True)
        criterion = nn.MSELoss()
        best_val_loss = float("inf")
        patience_counter = 0
        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            for batch in train_loader:
                states = batch["states"].to(device)
                target = batch["target"].to(device)
                optimizer.zero_grad()
                pred = self.forward(states)
                loss = criterion(pred, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * states.size(0)
            epoch_loss /= len(train_loader.dataset)

            if val_loader is not None:
                self.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        states = batch["states"].to(device)
                        target = batch["target"].to(device)
                        pred = self.forward(states)
                        val_loss += criterion(pred, target).item() * states.size(0)
                val_loss /= len(val_loader.dataset)
                scheduler.step(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        return self

__all__ = ["QuantumRegression__gen324", "RegressionDataset", "generate_superposition_data"]
