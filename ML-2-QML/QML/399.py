"""Quantum module producing embeddings for fusion with classical model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class QuantumNATEnhanced(tq.QuantumModule):
    """Quantum circuit generating a feature vector for hybrid fusion."""
    class QLayer(tq.QuantumModule):
        def __init__(self, depth: int = 2):
            super().__init__()
            self.depth = depth
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(
                n_ops=20 * depth, wires=list(range(self.n_wires))
            )
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.cnot = tq.Cnot
            self.hadamard = tq.Hadamard

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for _ in range(self.depth):
                self.rx(qdev, wires=0)
                self.ry(qdev, wires=1)
                self.rz(qdev, wires=2)
                self.cnot(qdev, wires=[0, 3])
                self.hadamard(qdev, wires=3)

    def __init__(self, embedding_dim: int = 16):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict["4x4_ryzxy"]
        )
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.embedding_dim = embedding_dim
        self.norm = nn.BatchNorm1d(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True
        )
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        out = self.norm(out)
        return out


__all__ = ["QuantumNATEnhanced"]
