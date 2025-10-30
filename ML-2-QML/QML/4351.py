"""Hybrid quantum model that mirrors the classical HybridNATModel
using TorchQuantum primitives.  It encodes images, applies a variational
layer, evaluates a kernel against a memory bank, and produces
classification logits, while also providing sampler and regression
outputs on separate quantum devices.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from torchquantum.functional import func_name_dict


class KernalAnsatz(tq.QuantumModule):
    """Encodes classical data through a programmable list of quantum gates."""

    def __init__(self, func_list):
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
    """Quantum kernel evaluated via a fixed TorchQuantum ansatz."""

    def __init__(self):
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


class QLayer(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
        self.rx0 = tq.RX(has_params=True, trainable=True)
        self.ry0 = tq.RY(has_params=True, trainable=True)
        self.rz0 = tq.RZ(has_params=True, trainable=True)
        self.crx0 = tq.CRX(has_params=True, trainable=True)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice):
        self.random_layer(qdev)
        self.rx0(qdev, wires=0)
        self.ry0(qdev, wires=1)
        self.rz0(qdev, wires=3)
        self.crx0(qdev, wires=[0, 2])
        tqf.hadamard(qdev, wires=3)
        tqf.sx(qdev, wires=2)
        tqf.cnot(qdev, wires=[3, 0])


class HybridNATModel(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

        # Sampler parameters
        self.sampler_weights = nn.Parameter(torch.randn(4))

        # Estimator parameter
        self.estimator_weight = nn.Parameter(torch.randn(1))

        # Memory bank for kernel
        self.memory = torch.randn(10, 4)

        # Kernel module
        self.kernel = Kernel()

    def _run_sampler(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=2, bsz=bsz, device=x.device)
        # Input rotations
        tq.RY(has_params=True, trainable=True)(qdev, wires=0, params=x[:, 0])
        tq.RY(has_params=True, trainable=True)(qdev, wires=1, params=x[:, 1])
        tq.CX(wires=[0, 1])(qdev)
        # Weight rotations
        tq.RY(has_params=True, trainable=True)(qdev, wires=0, params=self.sampler_weights[0])
        tq.RY(has_params=True, trainable=True)(qdev, wires=1, params=self.sampler_weights[1])
        tq.CX(wires=[0, 1])(qdev)
        tq.RY(has_params=True, trainable=True)(qdev, wires=0, params=self.sampler_weights[2])
        tq.RY(has_params=True, trainable=True)(qdev, wires=1, params=self.sampler_weights[3])
        probs = tq.MeasureAll(tq.PauliZ)(qdev)
        return probs

    def _run_estimator(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=1, bsz=bsz, device=x.device)
        tq.H()(qdev, wires=0)
        tq.RY(has_params=True, trainable=True)(qdev, wires=0, params=x[:, 0])
        tq.RX(has_params=True, trainable=True)(qdev, wires=0, params=self.estimator_weight)
        exp_y = tqf.expectation(tq.PauliY)(qdev)
        return exp_y

    def _kernel_matrix(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        M = memory.shape[0]
        mat = torch.zeros(bsz, M, device=x.device)
        for i in range(bsz):
            for j in range(M):
                mat[i, j] = self.kernel(x[i, :4].unsqueeze(0), memory[j].unsqueeze(0)).item()
        return mat

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz = x.shape[0]
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)

        # Quantum encoding and classification
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        logits = self.norm(out[:, :4])

        # Kernel similarity
        memory = self.memory.to(x.device)
        kernel_matrix = self._kernel_matrix(pooled, memory)

        # Sampler and estimator
        sampler_probs = self._run_sampler(pooled[:, :2])
        estimator_output = self._run_estimator(pooled[:, :1])

        return logits, sampler_probs, estimator_output, kernel_matrix


__all__ = ["HybridNATModel"]
