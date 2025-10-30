"""Quantum regression model with entangling ansatz and multi‑operator feature extraction."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq

def generate_superposition_data(num_wires: int, samples: int, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate states of the form cos(theta)|0...0> + exp(i phi) sin(theta)|1...1>.
    The target is a non‑linear function of theta and phi.
    Parameters
    ----------
    num_wires : int
        Number of qubits in each state.
    samples : int
        Number of samples to generate.
    seed : int | None, optional
        Random seed for reproducibility.
    Returns
    -------
    states, labels : np.ndarray
        State vectors (complex) and target scalars.
    """
    rng = np.random.default_rng(seed)
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * rng.random(samples)
    phis = 2 * np.pi * rng.random(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    """
    Torch dataset that returns quantum state tensors and scalar targets.
    """
    def __init__(self, samples: int, num_wires: int, seed: int | None = None):
        self.states, self.labels = generate_superposition_data(num_wires, samples, seed)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QModel(tq.QuantumModule):
    """
    Variational quantum circuit for regression with entangling layers and multi‑operator readout.
    The ansatz consists of:
        * a data‑encoding layer (GeneralEncoder)
        * several RandomLayers interleaved with CX entanglement
        * parameterised rotations on each qubit
    The measurement extracts expectation values of Pauli‑Z and Pauli‑X on all qubits,
    concatenated into a feature vector fed to a classical head.
    """
    class QLayer(tq.QuantumModule):
        """
        Sub‑module implementing one entangling block.
        """
        def __init__(self, num_wires: int):
            super().__init__()
            self.num_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.cx = tq.CX()
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            # Entangle all pairs with CX gates
            for i in range(self.num_wires - 1):
                self.cx(qdev, wires=[i, i + 1])
            for wire in range(self.num_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, num_wires: int, depth: int = 3):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.layers = nn.ModuleList([self.QLayer(num_wires) for _ in range(depth)])
        self.measure_z = tq.MeasureAll(tq.PauliZ)
        self.measure_x = tq.MeasureAll(tq.PauliX)
        self.head = nn.Linear(2 * num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state_batch : torch.Tensor
            Batch of complex state vectors (batch_size, 2**n_wires).
        Returns
        -------
        torch.Tensor
            Predicted scalar values (batch_size).
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        for layer in self.layers:
            layer(qdev)
        z_features = self.measure_z(qdev)
        x_features = self.measure_x(qdev)
        features = torch.cat([z_features, x_features], dim=-1)
        return self.head(features).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
