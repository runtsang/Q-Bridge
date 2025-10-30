"""
HybridKernelRegression – quantum implementation.

This module mirrors the classical API but replaces the RBF kernel with a
parameter‑free quantum kernel based on TorchQuantum.  Data is encoded
into a 4‑qubit device with a simple Ry rotation per qubit, followed
by a reversed encoding that implements the inner‑product trick.
A hybrid quantum‑classical head then maps the measured expectation
values to a scalar output.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum.functional import func_name_dict

# ----------------------------------------------------------------------
# Quantum kernel primitives
# ----------------------------------------------------------------------


class QuantumKernalAnsatz(tq.QuantumModule):
    """Encodes two classical vectors using a reversible circuit."""

    def __init__(self, func_list: list[dict]) -> None:
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = (
                x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            )
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = (
                -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            )
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)


class QuantumKernel(tq.QuantumModule):
    """Fixed 4‑qubit quantum kernel."""

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = QuantumKernalAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Compute the quantum Gram matrix."""
    kernel = QuantumKernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])


# ----------------------------------------------------------------------
# Dataset utilities (identical interface to the classical version)
# ----------------------------------------------------------------------


def generate_quantum_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample superposition states |0…0> + e^{iφ} |1…1>."""
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
    """Dataset that returns complex quantum states and scalar targets."""

    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_quantum_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


# ----------------------------------------------------------------------
# Quantum hybrid model
# ----------------------------------------------------------------------


class HybridKernelRegression(tq.QuantumModule):
    """
    Quantum hybrid kernel regression.

    The model encodes input states into a 4‑qubit device, measures all
    qubits in the Pauli‑Z basis, and applies a classical linear head.
    """

    class QLayer(tq.QuantumModule):
        """Random layer + trainable single‑qubit rotations."""

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

    def __init__(self, num_wires: int):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{num_wires}xRy"]
        )
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """Evaluate the hybrid model on a batch of quantum states."""
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)


__all__ = [
    "QuantumKernalAnsatz",
    "QuantumKernel",
    "kernel_matrix",
    "generate_quantum_data",
    "RegressionDataset",
    "HybridKernelRegression",
]
