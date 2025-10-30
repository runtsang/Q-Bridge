"""Quantum regression dataset and model with tunable entanglement depth."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from typing import Tuple

def generate_superposition_data(num_wires: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate superposition states |ψ> = cosθ|0...0> + e^{iφ}sinθ|1...1>.

    Parameters
    ----------
    num_wires : int
        Number of qubits.
    samples : int
        Number of samples.

    Returns
    -------
    states : np.ndarray shape (samples, 2**num_wires)
        Complex state vectors.
    labels : np.ndarray shape (samples,)
        Regression targets.
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
    """Dataset for the quantum regression problem."""

    def __init__(self, samples: int, num_wires: int):
        super().__init__()
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QModel(tq.QuantumModule):
    """Quantum regression model with a variational block and tunable entanglement depth.

    Parameters
    ----------
    num_wires : int
        Number of qubits.
    depth : int, default 3
        Number of layers in the variational block.
    entangler_type : str, default "cx"
        Entanglement gate type; can be 'cx', 'cz', or'swap'.
    """

    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int, depth: int, entangler_type: str):
            super().__init__()
            self.n_wires = num_wires
            self.depth = depth
            self.entangler = tq.EntanglementLayer(
                wires=list(range(num_wires)),
                depth=depth,
                entangler_type=entangler_type,
                has_params=True,
                trainable=True,
            )
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.entangler(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, num_wires: int, depth: int = 3, entangler_type: str = "cx"):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires, depth, entangler_type)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
