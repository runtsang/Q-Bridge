"""Quantum regression model with richer variational circuit.

The QML implementation expands the original architecture by adding an
entangling layer, measuring multiple Pauli observables, and a
classical readout head that operates on the concatenated measurement
vector.  The encoder now supports both amplitude and phase encoding to
better capture the structure of the synthetic data.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Create states |ψ(θ,φ)⟩ = cosθ|0⋯0⟩ + e^{iφ} sinθ |1⋯1⟩.

    Parameters
    ----------
    num_wires : int
        Number of qubits in the device.
    samples : int
        Number of training examples.

    Returns
    -------
    states, labels : tuple[np.ndarray, np.ndarray]
        ``states`` has shape ``(samples, 2**num_wires)`` and contains
        complex amplitudes. ``labels`` are a nonlinear function of the
        underlying angles.
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
    """Dataset returning quantum states and scalar targets."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class QModel(tq.QuantumModule):
    """
    Variational quantum regression model.

    The circuit consists of:
        * An amplitude+phase encoder (Ry followed by Rz on each wire).
        * A randomized layer of 30 two‑qubit gates.
        * A trainable single‑qubit rotation layer (RX+RY) on each wire.
        * An entangling layer of CX gates in a ring topology.
    Measurements of both Pauli‑Z and Pauli‑X are concatenated and fed
    into a small classical linear layer.
    """

    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
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

    class EntangleRing(tq.QuantumModule):
        """Entangle neighboring wires with CX in a ring."""
        def __init__(self, num_wires: int):
            super().__init__()
            self.cx = tq.CNOT
            self.num_wires = num_wires

        def forward(self, qdev: tq.QuantumDevice) -> None:
            for i in range(self.num_wires):
                self.cx(qdev, wires=[i, (i + 1) % self.num_wires])

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        # Encoder uses a built‑in amplitude+phase mapping
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRyRz"])
        self.q_layer = self.QLayer(num_wires)
        self.entangle = self.EntangleRing(num_wires)
        # Measure both Z and X
        self.measure_z = tq.MeasureAll(tq.PauliZ)
        self.measure_x = tq.MeasureAll(tq.PauliX)
        self.head = nn.Linear(2 * num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        self.entangle(qdev)
        z_feat = self.measure_z(qdev)
        x_feat = self.measure_x(qdev)
        features = torch.cat([z_feat, x_feat], dim=1)  # shape: (bsz, 2*num_wires)
        return self.head(features).squeeze(-1)


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
