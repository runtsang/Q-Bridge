"""Advanced quantum regression model with flexible encoder and ansatz."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample states of the form cos(theta)|0...0> + e^{i phi} sin(theta)|1...1>.
    The label is a smooth function of theta and phi.
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
    """Dataset that returns quantum state tensors and scalar targets."""

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
    Quantum regression model with a configurable encoder, entangling ansatz,
    and classical post‑processing head.

    Parameters
    ----------
    num_wires : int
        Number of qubits in the device.
    encoder_name : str, optional
        Name of the built‑in encoder to use. Defaults to f"{num_wires}xRy".
    n_ansatz_layers : int, optional
        Number of variational layers. Defaults to 3.
    """

    class EntanglingLayer(tq.QuantumModule):
        """Layer consisting of a random unitary followed by parameterized rotations."""

        def __init__(self, num_wires: int):
            super().__init__()
            self.random = tq.RandomLayer(n_ops=20, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random(qdev)
            for wire in range(qdev.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, num_wires: int, encoder_name: str = None, n_ansatz_layers: int = 3):
        super().__init__()
        self.n_wires = num_wires
        encoder_name = encoder_name or f"{num_wires}xRy"
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[encoder_name])
        self.ansatz = tq.Sequential(
            *[self.EntanglingLayer(num_wires) for _ in range(n_ansatz_layers)]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.ansatz(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
