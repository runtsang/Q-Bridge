"""Hybrid quantum‑classical model for classification and regression.

The quantum implementation mirrors the classical architecture but replaces
the random‑weight emulator with a variational circuit.  The encoder maps
classical CNN features onto a 4‑qubit register; a random layer followed by
parameterised RX/RY/RZ/CRX gates acts as the variational head.  Measurement
produces a 4‑dimensional feature vector that is optionally fed to a linear
regression head.  The design demonstrates how a quantum circuit can be
integrated into a conventional CNN pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np
from torch.utils.data import Dataset


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate quantum superposition states and corresponding labels.

    Parameters
    ----------
    num_wires : int
        Number of qubits in the state.
    samples : int
        Number of samples.

    Returns
    -------
    states : np.ndarray
        Complex state vectors of shape (samples, 2**num_wires).
    labels : np.ndarray
        Real labels of shape (samples,).
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
    return states, labels


class RegressionDataset(torch.utils.data.Dataset):
    """Dataset wrapping quantum superposition data for regression."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class HybridNATModel(tq.QuantumModule):
    """Quantum hybrid model combining CNN feature extraction, a variational circuit,
    and a classification/regression head.

    Parameters
    ----------
    regression : bool, default False
        If True, the network outputs a scalar regression value.
        Otherwise, it outputs a 4‑dimensional classification vector.
    """

    class QLayer(tq.QuantumModule):
        """Variational circuit used as the quantum head."""

        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(n_wires)))
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self, regression: bool = False) -> None:
        super().__init__()
        self.n_wires = 4
        # Encoder that maps the 16‑dimensional pooled CNN feature vector onto 4 qubits.
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4xRy"])
        self.q_layer = self.QLayer(self.n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)
        self.regression = regression
        if self.regression:
            self.head = nn.Linear(self.n_wires, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True
        )
        # Classical CNN feature extractor
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        features = self.measure(qdev)
        if self.regression:
            out = self.head(features).squeeze(-1)
        else:
            out = self.norm(features)
        return out


__all__ = ["HybridNATModel", "RegressionDataset", "generate_superposition_data"]
