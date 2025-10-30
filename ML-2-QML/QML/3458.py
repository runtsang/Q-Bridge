"""Quantum regression model with a variational circuit and a classical readout layer.

This module keeps the original functionality from the seed but introduces a
slightly more expressive encoder and a trainable readout network.  All
components are built with ``torchquantum`` so the model can be trained with
gradient‑based optimisers.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq


def generate_superposition_data(
    num_wires: int,
    samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample states of the form cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>.
    The labels are a non‑linear function of the angles.
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


class RegressionDataset(torch.utils.data.Dataset):
    """
    Dataset that returns a quantum state and a regression target.
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


class QuantumFeatureExtractor(tq.QuantumModule):
    """
    A variational circuit that encodes a classical vector into a quantum state
    and returns Pauli‑Z expectation values.
    """

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{num_wires}xRy"],
            parametric=True,
        )
        self.random_layer = tq.RandomLayer(
            n_ops=30,
            wires=list(range(num_wires)),
            has_params=True,
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.encoder(qdev)
        self.random_layer(qdev)
        return self.measure(qdev)


class QModel(tq.QuantumModule):
    """
    Full quantum regression model with a classical readout head.
    """

    def __init__(self, num_wires: int, hidden_sizes: list[int] = [32, 16]):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = QuantumFeatureExtractor(num_wires)

        # Classical head
        layers = []
        in_dim = num_wires
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode, measure, and read out with a classical network.

        Parameters
        ----------
        state_batch : torch.Tensor, shape (batch, 2**num_wires)

        Returns
        -------
        preds : torch.Tensor, shape (batch,)
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires,
            bsz=bsz,
            device=state_batch.device,
        )
        qdev.state = state_batch.to(tq.device)
        features = self.encoder(qdev)  # shape (batch, num_wires)
        preds = self.head(features).squeeze(-1)
        return preds


__all__ = ["generate_superposition_data", "RegressionDataset", "QuantumFeatureExtractor", "QModel"]
