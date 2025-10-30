"""Hybrid quantum model combining a quantum encoder, variational layer, and quantum kernel classification head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from torchquantum.functional import func_name_dict


class QFCModel(tq.QuantumModule):
    """
    Quantum hybrid model:
    - General encoder mapping classical data to a 4-qubit state
    - Variational layer with random and trainable gates
    - Optional quantum kernel head using a fixed ansatz
    - Measurement in PauliZ basis and batch normalization
    """

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

    class QuantumKernel(tq.QuantumModule):
        """
        Fixed ansatz for quantum kernel evaluation.
        """
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
            self.ansatz = [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor):
            qdev.reset_states(x.shape[0])
            for info in self.ansatz:
                params = x[:, info["input_idx"]] if func_name_dict[info["func"]].num_params else None
                func_name_dict[info["func"]](qdev, wires=info["wires"], params=params)
            for info in reversed(self.ansatz):
                params = -y[:, info["input_idx"]] if func_name_dict[info["func"]].num_params else None
                func_name_dict[info["func"]](qdev, wires=info["wires"], params=params)

        def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            self.forward(self.q_device, x, y)
            return torch.abs(self.q_device.states.view(-1)[0])

    def __init__(self, use_kernel: bool = True):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)
        self.use_kernel = use_kernel
        if self.use_kernel:
            self.kernel_head = self.QuantumKernel()
            self.prototypes = nn.Parameter(torch.randn(4, self.n_wires))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing logits for classification.
        """
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        pooled = torch.nn.functional.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        out = self.norm(out)

        if self.use_kernel:
            sims = torch.stack(
                [self.kernel_head.evaluate(out, proto.unsqueeze(0)) for proto in self.prototypes],
                dim=1,
            )
            logits = sims
        else:
            logits = out

        return logits
