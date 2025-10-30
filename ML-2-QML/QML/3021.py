"""Hybrid regression/classification quantum module, extending the original QuantumRegression seed."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torch.utils.data import Dataset
from typing import Iterable, Tuple

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample states of the form cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>.
    Returns complex state vectors, regression labels and binary classification labels.
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
    y_reg = np.sin(2 * thetas) * np.cos(phis)
    y_cls = (y_reg >= 0).astype(np.float32)
    return states, y_reg.astype(np.float32), y_cls

class HybridQuantumDataset(Dataset):
    """
    Quantum dataset providing state vectors, regression targets and binary classification targets.
    """
    def __init__(self, samples: int, num_wires: int):
        self.states, self.regression_targets, self.classification_targets = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "regression_target": torch.tensor(self.regression_targets[index], dtype=torch.float32),
            "classification_target": torch.tensor(self.classification_targets[index], dtype=torch.float32),
        }

class HybridQuantumModel(tq.QuantumModule):
    """
    Quantum variational circuit with shared feature extraction and dual classical heads.
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

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.regression_head = nn.Linear(num_wires, 1)
        self.classification_head = nn.Linear(num_wires, 2)

    def forward(self, state_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        regression_output = self.regression_head(features).squeeze(-1)
        classification_output = self.classification_head(features)
        return regression_output, classification_output

# Backwards‑compatibility aliases
RegressionDataset = HybridQuantumDataset
QModel = HybridQuantumModel

__all__ = ["HybridQuantumModel", "HybridQuantumDataset", "RegressionDataset", "QModel", "generate_superposition_data"]
