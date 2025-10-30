"""Quantum regression model with a configurable encoder and hybrid loss."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torch.utils.data import Dataset, random_split

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample states of the form cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>.
    """
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
    """
    Dataset that returns quantum state vectors and target scalars.
    """
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

def split_dataset(dataset: Dataset, val_ratio: float = 0.2) -> tuple[Dataset, Dataset]:
    """
    Split a dataset into training and validation subsets.
    """
    n_total = len(dataset)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val
    return random_split(dataset, [n_train, n_val])

class QModelHybrid(tq.QuantumModule):
    """
    Quantum regression model with a selectable encoder.
    The encoder can be 'ry', 'rx', or 'cz' (default is 'ry').
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, num_wires: int, encoder_name: str = "ry"):
        super().__init__()
        self.n_wires = num_wires
        # Map encoder names to predefined gate lists
        encoder_dict = {
            "ry": tq.encoder_op_list_name_dict[f"{num_wires}xRy"],
            "rx": tq.encoder_op_list_name_dict[f"{num_wires}xRx"],
            "cz": tq.encoder_op_list_name_dict[f"{num_wires}xCZ"],
        }
        self.encoder = tq.GeneralEncoder(encoder_dict.get(encoder_name, tq.encoder_op_list_name_dict[f"{num_wires}xRy"]))
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

    @staticmethod
    def hybrid_loss(classical_pred: torch.Tensor,
                    quantum_pred: torch.Tensor,
                    target: torch.Tensor,
                    alpha: float = 0.5) -> torch.Tensor:
        """
        Weighted mean‑square error between predictions and target.
        """
        mse = nn.MSELoss()
        return alpha * mse(classical_pred, target) + (1.0 - alpha) * mse(quantum_pred, target)

__all__ = ["QModelHybrid", "RegressionDataset", "generate_superposition_data", "split_dataset"]
