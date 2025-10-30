"""
RegressionModel: Quantum‑classical hybrid regression with configurable encoder
and variational circuit.
"""

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torch.utils.data import DataLoader

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate amplitude‑encoded states and regression targets."""
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
    """Dataset returning amplitude‑encoded states and labels."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int):
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class RegressionModel(tq.QuantumModule):
    """Hybrid regression model with a random entangling layer and trainable rotations."""
    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)

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
        *,
        epochs: int = 100,
        lr: float = 1e-3,
        val_loader: DataLoader | None = None,
        patience: int = 10,
        device: torch.device | str | None = None,
    ) -> None:
        """Train the hybrid model with early stopping."""
        device = torch.device(device or "cpu")
        self.to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        best_val = float("inf")
        bad_epochs = 0

        for epoch in range(epochs):
            self.train()
            for batch in train_loader:
                optimizer.zero_grad()
                preds = self(batch["states"].to(device))
                loss = criterion(preds, batch["target"].to(device))
                loss.backward()
                optimizer.step()

            if val_loader is not None:
                self.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        preds = self(batch["states"].to(device))
                        val_loss += criterion(preds, batch["target"].to(device)).item()
                val_loss /= len(val_loader)
                if val_loss < best_val:
                    best_val = val_loss
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                if bad_epochs >= patience:
                    break

    def predict(self, state_batch: torch.Tensor, *, device: torch.device | str | None = None):
        """Convenience prediction wrapper."""
        device = torch.device(device or "cpu")
        self.eval()
        with torch.no_grad():
            return self(state_batch.to(device)).cpu()

__all__ = ["RegressionModel", "RegressionDataset", "generate_superposition_data"]
