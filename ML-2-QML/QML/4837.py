"""Quantum regression model that augments the classical CNN with a variational quantum layer
and a measurement‑based head.  The quantum module re‑uses the data encoder,
randomised gate layer, and measurement strategy from the original
QuantumRegression and QuantumNAT examples, while adding a small
quantum kernel for similarity assessment.

The class name matches the classical counterpart (QModel) so that
experiments can switch back‑and‑forth between backends seamlessly.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum.functional import func_name_dict
from torchquantum import tqf


def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset of superposition states and corresponding labels
    using a cosine/sine mixture, mirroring the classical generator.
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
    Dataset wrapper for the quantum superposition generator.
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


class KernalAnsatz(tq.QuantumModule):
    """
    Quantum kernel ansatz that encodes two classical vectors into a single device
    and returns the overlap amplitude.
    """
    def __init__(self, func_list: list[dict]) -> None:
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)


class Kernel(tq.QuantumModule):
    """
    Quantum kernel module that evaluates a fixed 4‑wire ansatz.
    """
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(
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


class QModel(tq.QuantumModule):
    """
    Hybrid quantum regression model that combines a quantum encoder,
    a variational quantum layer, and a classical linear head.
    The CNN feature extractor from the classical side is emulated by the
    GeneralEncoder, while the measurement and head mirror the original
    QuantumRegression architecture.
    """
    class QLayer(tq.QuantumModule):
        """
        Variational layer implementing a randomised circuit of RX/RY/RZ/CRX,
        followed by a fixed sequence of Hadamard, SX, and CNOT gates.
        """
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.crx = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)
                self.rz(qdev, wires=wire)
                self.crx(qdev, wires=[wire, (wire + 1) % self.n_wires])
            tqf.hadamard(qdev, wires=0, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=1, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[2, 3], static=self.static_mode, parent_graph=self.graph)

    def __init__(self, num_wires: int = 4):
        super().__init__()
        self.n_wires = num_wires
        # Encoder maps classical data to a quantum state using a fixed Ry‑Z‑X‑Y pattern
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)
        self.norm = nn.BatchNorm1d(num_wires)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        # Encode the classical batch into the device
        self.encoder(qdev, state_batch)
        # Apply the variational layer
        self.q_layer(qdev)
        # Measure expectation values
        features = self.measure(qdev)
        # Classical post‑processing
        out = self.head(features)
        return self.norm(out).squeeze(-1)


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data", "Kernel", "KernalAnsatz"]
