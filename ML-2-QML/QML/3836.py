"""Quantum version of HybridNATRegression using TorchQuantum."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np


# ------------------------------------------- #
#   Quantum data generation
# ------------------------------------------- #
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate states of the form |ψ⟩ = cos(θ)|0…0⟩ + e^{iφ} sin(θ)|1…1⟩
    and corresponding labels similar to the classical regression target.
    """
    omega0 = np.zeros(2 ** num_wires, dtype=complex)
    omega0[0] = 1.0
    omega1 = np.zeros(2 ** num_wires, dtype=complex)
    omega1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels


class RegressionDataset(torch.utils.data.Dataset):
    """Quantum dataset mirroring the classical regression dataset."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


# ------------------------------------------- #
#   Quantum hybrid model
# ------------------------------------------- #
class HybridNATRegression(tq.QuantumModule):
    """
    Quantum counterpart of HybridNATRegression.  A classical image encoder
    followed by a parameterised quantum circuit acts as the feature extractor.
    Two measurement heads provide classification logits and a regression output.
    """
    class QLayer(tq.QuantumModule):
        """
        Parameterised quantum layer that applies random rotations to each wire
        and a small entangling block to capture correlations.
        """
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=40, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)
            # Entangle neighbouring wires
            for w in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[w, w + 1], static=self.static_mode, parent_graph=self.graph)

    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires

        # Classical encoder: identical to the ML version
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict["4xRy"] if n_wires == 4 else tq.encoder_op_list_name_dict[f"{n_wires}xRy"]
        )

        # Quantum feature extractor
        self.q_layer = self.QLayer(n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Classical heads for classification and regression
        self.cls_head = nn.Sequential(
            nn.Linear(n_wires, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 4),
            nn.BatchNorm1d(4),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(n_wires, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state_batch: Tensor of shape [batch, 2**n_wires] containing complex amplitudes.
        Returns:
            logits: [batch, 4] classification logits
            pred:   [batch] regression prediction
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device, record_op=True)

        # Encode classical information (here we simply use the raw amplitudes)
        self.encoder(qdev, state_batch)

        # Quantum layer
        self.q_layer(qdev)

        # Measurement
        features = self.measure(qdev)

        logits = self.cls_head(features)
        pred = self.reg_head(features).squeeze(-1)
        return logits, pred


__all__ = ["HybridNATRegression", "RegressionDataset", "generate_superposition_data"]
