"""Quantum hybrid model integrating a quantum encoder, a variational layer, and a quantum kernel."""

from __future__ import annotations

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict, op_name_dict
import torchquantum.functional as tqf

class QuantumKernelAnsatz(tq.QuantumModule):
    """Encodes two classical inputs into a quantum device and computes the overlap."""
    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class QuantumLayer(tq.QuantumModule):
    """Variational layer inspired by Quantum‑NAT."""
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
        tqf.hadamard(qdev, wires=3)
        tqf.sx(qdev, wires=2)
        tqf.cnot(qdev, wires=[3, 0])

class HybridKernelNAT(tq.QuantumModule):
    """Quantum hybrid model integrating a quantum encoder, a variational layer, and a quantum kernel."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        # Quantum device used for feature encoding
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        # Encoder: 4x4_ryzxy pattern
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        # Variational layer
        self.q_layer = QuantumLayer()
        # Quantum kernel ansatz
        self.kernel_ansatz = QuantumKernelAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = torch.nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the quantum kernel between two batches of classical data."""
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        # Feature map: average pool and flatten
        pooled = torch.nn.functional.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        # Kernel evaluation
        self.kernel_ansatz(self.q_device, x, y)
        kernel_val = torch.abs(self.q_device.states.view(-1)[0])
        return self.norm(out), kernel_val

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
        """Return Gram matrix computed by the quantum kernel."""
        return np.array([[self.forward(x, y)[1].item() for y in b] for x in a])

__all__ = ["HybridKernelNAT"]
