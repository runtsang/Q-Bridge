"""Hybrid quantum regression model based on the original seed.

The quantum variant encodes the classical data into a variational circuit and
extracts observables that are passed to a linear head.  The model is fully
differentiable and can be trained with standard optimisers.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- #
# Dataset utilities – identical to the original seed
# --------------------------------------------------------------------------- #
def generate_superposition_data(
    num_wires: int, samples: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sample states of the form cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>."""
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = (
            np.cos(thetas[i]) * omega_0
            + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
        )
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels


class RegressionDataset(Dataset):
    """Simple torch Dataset for the synthetic quantum regression data."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


# --------------------------------------------------------------------------- #
# Hybrid quantum regression model
# --------------------------------------------------------------------------- #
class HybridRegressionModel(tq.QuantumModule):
    """Quantum variational circuit with a linear read‑out."""

    class QLayer(tq.QuantumModule):
        """Variational layer with random gates followed by parametrised rotations."""

        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(
                n_ops=30, wires=list(range(num_wires))
            )
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(
        self,
        num_wires: int,
        output_dim: int = 1,
        *,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.n_wires = num_wires
        # Parametrised encoder that maps classical data onto the quantum device
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{num_wires}xRy"]
        )
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, output_dim)
        self.to(device)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """Encode the batch, run the variational circuit, and read out."""
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=bsz, device=state_batch.device
        )
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)


__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data"]
