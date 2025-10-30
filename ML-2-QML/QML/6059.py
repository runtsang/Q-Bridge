"""Quantum implementation with 8‑qubit variational encoder and projection head."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class QuantumNATEnhanced(tq.QuantumModule):
    """Variational model inspired by Quantum‑NAT, expanded to 8 qubits."""

    class VariationalLayer(tq.QuantumModule):
        """Trainable variational block that mixes all wires."""

        def __init__(self):
            super().__init__()
            self.n_wires = 8
            self.random = tq.RandomLayer(n_ops=80, wires=list(range(self.n_wires)))
            self.encoder = tq.GeneralEncoder(
                tq.encoder_op_list_name_dict["8x8_ryzxy"]
            )

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random(qdev)
            self.encoder(qdev)

    def __init__(self):
        super().__init__()
        self.n_wires = 8
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict["8x8_ryzxy"]
        )
        self.var_layer = self.VariationalLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)
        self.proj = nn.Linear(self.n_wires, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True
        )
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.var_layer(qdev)
        out = self.measure(qdev)
        out = self.norm(out)
        out = self.proj(out)
        return out


__all__ = ["QuantumNATEnhanced"]
