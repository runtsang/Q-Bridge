"""Hybrid quantum‑classical regression model.

The module contains:
* `generate_superposition_data` – identical to the ML counterpart for consistency.
* `RegressionDataset` – same PyTorch Dataset yielding complex states and targets.
* `HybridModel` – torchquantum module that encodes the state, applies a random layer and
  parameterised RX/RY gates, measures Pauli‑Z, and projects to a single‑dimensional
  feature vector.  This vector is meant to be concatenated with the classical output
  of `HybridModel` defined in the ML module.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torch.utils.data import Dataset


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data: a superposition of |0…0⟩ and |1…1⟩ with random angles.

    The returned *features* are the complex amplitudes of the quantum states,
    while *labels* are a smooth function of the underlying angles.
    """
    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)

    # Build the two basis states
    omega_0 = np.zeros(2 ** num_features, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_features, dtype=complex)
    omega_1[-1] = 1.0

    states = np.cos(thetas[:, None]) * omega_0 + np.exp(1j * phis[:, None]) * np.sin(thetas[:, None]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)

    return states.astype(np.complex64), labels.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset that yields a complex state tensor and the regression target."""

    def __init__(self, samples: int, num_features: int):
        self.states, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class HybridModel(tq.QuantumModule):
    """Quantum component of the hybrid regression model.

    The circuit encodes the input state via a general Ry‑encoder, applies a random
    layer followed by parameterised RX/RY gates, measures Pauli‑Z on all wires,
    and projects the measurement vector to a single‑dimensional feature via a
    classical linear layer.  The output is intended to be concatenated with the
    classical stream of the `HybridModel` defined in the ML module.
    """

    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state_batch : torch.Tensor
            Complex tensor of shape (batch, 2**num_wires).

        Returns
        -------
        torch.Tensor
            Quantum feature vector of shape (batch, 1).
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)  # (bsz, n_wires)
        return self.head(features).squeeze(-1)


__all__ = ["HybridModel", "RegressionDataset", "generate_superposition_data"]
