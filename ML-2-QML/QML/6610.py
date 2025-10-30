"""Quantum regression module with configurable encoders and hybrid head.

The module defines:
    * `generate_superposition_data` – produces superposition states and labels.
    * `RegressionDataset` – torch Dataset yielding complex states and targets.
    * `QuantumRegression__gen456` – a tq.QuantumModule that encodes the input, applies a variational layer, measures, and feeds the result into a classical head.
    * `evaluate_cross_validation` – helper that runs k‑fold CV on a DataLoader.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
from typing import Tuple, List

def generate_superposition_data(num_wires: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sample states of the form cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>."""
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

class RegressionDataset(Dataset):
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QuantumRegression__gen456(tq.QuantumModule):
    def __init__(
        self,
        num_wires: int,
        encoder_name: str = "Ry",
        variational_layers: int = 2,
        hidden_units: int = 32,
    ):
        super().__init__()
        self.n_wires = num_wires
        # configurable encoder
        encoder_dict = tq.encoder_op_list_name_dict
        self.encoder = tq.GeneralEncoder(encoder_dict[f"{num_wires}x{encoder_name}"])
        # variational block
        self.variational = tq.RandomLayer(n_ops=30 * variational_layers, wires=list(range(num_wires)))
        # measurement
        self.measure = tq.MeasureAll(tq.PauliZ)
        # classical head
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.variational(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

    def evaluate_cross_validation(
        self,
        loader: DataLoader,
        k_folds: int = 5,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[float, float]:
        """Return mean MSE and accuracy over k‑fold cross‑validation."""
        dataset = loader.dataset
        indices = np.arange(len(dataset))
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        mse_scores: List[float] = []
        acc_scores: List[float] = []

        for train_idx, val_idx in kf.split(indices):
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            train_loader = DataLoader(train_subset, batch_size=loader.batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=loader.batch_size, shuffle=False)

            self.train()
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
            criterion = nn.MSELoss()

            for _ in range(10):  # few epochs per fold
                for batch in train_loader:
                    optimizer.zero_grad()
                    outputs = self(batch["states"].to(device))
                    loss = criterion(outputs, batch["target"].to(device))
                    loss.backward()
                    optimizer.step()

            self.eval()
            mse = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    outputs = self(batch["states"].to(device))
                    mse += criterion(outputs, batch["target"].to(device)).item() * batch["states"].size(0)
                    preds = (outputs > 0).long()
                    correct += (preds == (batch["target"].abs() > 0.5).long()).sum().item()
                    total += batch["states"].size(0)

            mse_scores.append(mse / total)
            acc_scores.append(correct / total)

        return float(np.mean(mse_scores)), float(np.mean(acc_scores))

__all__ = ["QuantumRegression__gen456", "RegressionDataset", "generate_superposition_data"]
