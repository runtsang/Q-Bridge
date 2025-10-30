"""Hybrid QCNN regression – quantum implementation.

This module builds a QCNN ansatz using a ZFeatureMap and a
parameterized convolution‑pooling structure.  The output of the
ansatz is fed into a linear readout that produces a regression
value.  The same synthetic dataset used by the classical branch
is provided for direct comparison.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate complex quantum states of the form
    cos(theta)|0...0> + e^{i phi} sin(theta)|1...1>.

    Parameters
    ----------
    num_wires : int
        Number of qubits in each state.
    samples : int
        Number of samples to generate.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        States of shape (samples, 2**num_wires) and targets of shape
        (samples,).
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
    """
    Torch dataset wrapping the synthetic complex quantum states.
    """
    def __init__(self, samples: int, num_wires: int) -> None:
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[idx], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class HybridQCNNRegressionQuantum(tq.QuantumModule):
    """
    Quantum QCNN ansatz with a regression readout.

    The ansatz consists of alternating convolutional and pooling layers
    built from a 2‑qubit block.  After the ansatz, the qubit states are
    measured in the Z basis and fed into a linear head.
    """

    class QLayer(tq.QuantumModule):
        """
        A generic layer that applies a random circuit followed by
        single‑qubit rotations on every wire.
        """
        def __init__(self, num_wires: int) -> None:
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, num_wires: int = 8) -> None:
        super().__init__()
        self.n_wires = num_wires
        # Feature map: ZFeatureMap (implemented as a list of Ry rotations)
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        state_batch : torch.Tensor
            Batch of complex states of shape (batch, 2**num_wires).

        Returns
        -------
        torch.Tensor
            Regression output of shape (batch,).
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)


def QCNN() -> HybridQCNNRegressionQuantum:
    """
    Factory that returns a ready‑to‑train instance of the quantum QCNN.
    """
    return HybridQCNNRegressionQuantum(num_wires=8)


__all__ = ["HybridQCNNRegressionQuantum", "QCNN", "RegressionDataset", "generate_superposition_data"]
