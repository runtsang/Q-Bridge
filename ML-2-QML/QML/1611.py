"""Quantum regression model with entangling layers and dual‑stage encoding."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate complex amplitudes of the form
    cos(theta)|0…0⟩ + e^{i phi} sin(theta)|1…1⟩.

    Parameters
    ----------
    num_wires : int
        Number of qubits.
    samples : int
        Number of samples to generate.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        State matrix (complex) and target values.
    """
    omega_0 = np.zeros(2**num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2**num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2**num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels


class RegressionDataset(torch.utils.data.Dataset):
    """
    Dataset that returns complex state tensors and real targets.
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


class QuantumRegressionGen177(tq.QuantumModule):
    """
    Variational quantum circuit with entangling depth and dual encoding.
    """

    class QLayer(tq.QuantumModule):
        """
        Parameterised layer that applies random gates, rotations,
        and a fixed entangling pattern.
        """

        def __init__(self, num_wires: int, depth: int = 2):
            super().__init__()
            self.n_wires = num_wires
            self.depth = depth
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.cnot = tq.CNOT

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for _ in range(self.depth):
                for wire in range(self.n_wires - 1):
                    self.cnot(qdev, control_wires=[wire], target_wires=[wire + 1])
                for wire in range(self.n_wires):
                    self.rx(qdev, wires=wire)
                    self.ry(qdev, wires=wire)

    def __init__(self, num_wires: int, qlayer_depth: int = 2):
        super().__init__()
        self.n_wires = num_wires
        self.encoder1 = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.encoder2 = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires, depth=qlayer_depth)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        state_batch : torch.Tensor
            Batch of complex states, shape (batch_size, 2**num_wires).

        Returns
        -------
        torch.Tensor
            Predicted values, shape (batch_size,).
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires,
            bsz=bsz,
            device=state_batch.device,
        )
        self.encoder1(qdev, state_batch)
        self.q_layer(qdev)
        self.encoder2(qdev, state_batch)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)


__all__ = ["QuantumRegressionGen177", "RegressionDataset", "generate_superposition_data"]
