"""Hybrid quantum kernel and quantum neural network model.

The quantum kernel uses a parameter‑free ansatz that encodes two
vectors in opposite directions.  The quantum fully‑connected block
mirrors the structure of the seed but adds a random layer for
expressivity.  The top‑level :class:`HybridKernelNATModel` stitches
them together, enabling a kernel‑based quantum‑feature extractor
followed by a quantum classifier.
"""

from __future__ import annotations

import torch
import torchquantum as tq
import torchquantum.functional as tqf
from torch import nn
import numpy as np
from typing import Sequence


class QuantumKernelAnsatz(tq.QuantumModule):
    """Quantum kernel that encodes two vectors with a reversible circuit."""
    def __init__(self, gate_list):
        super().__init__()
        self.gate_list = gate_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice,
                x: torch.Tensor, y: torch.Tensor) -> None:
        # Reset device states to the batch size of x.
        q_device.reset_states(x.shape[0])
        # Encode x.
        for info in self.gate_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        # Un‑encode y with inverse parameters.
        for info in reversed(self.gate_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)


class QuantumKernel(tq.QuantumModule):
    """Fixed‑depth quantum kernel using a 4‑qubit circuit."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = QuantumKernelAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Reshape to (1, -1) to match the ansatz expectations.
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        # Return the real part of the first state amplitude.
        return torch.abs(self.q_device.states.view(-1)[0])


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    kernel = QuantumKernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])


class QuantumQLayer(tq.QuantumModule):
    """Randomized layer followed by trainable gates."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
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


class QuantumQFCModel(tq.QuantumModule):
    """Quantum version of the CNN‑FC architecture."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = QuantumQLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    @tq.static_support
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz,
                                device=x.device, record_op=True)
        pooled = torch.nn.functional.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)


class HybridKernelNATModel(tq.QuantumModule):
    """Hybrid quantum kernel + quantum classifier."""
    def __init__(self,
                 kernel: tq.QuantumModule,
                 support: torch.Tensor,
                 model: tq.QuantumModule) -> None:
        super().__init__()
        self.kernel = kernel
        self.support = support
        self.model = model

    @tq.static_support
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = self.kernel(x, self.support)          # (batch, 4)
        k = k.view(x.shape[0], -1)
        return self.model(k)


__all__ = [
    "QuantumKernelAnsatz",
    "QuantumKernel",
    "kernel_matrix",
    "QuantumQLayer",
    "QuantumQFCModel",
    "HybridKernelNATModel",
]
