"""Quantum‑enhanced architecture for QuantumNAT with a parameterised variational circuit."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class QuantumNATEnhanced(tq.QuantumModule):
    """Quantum module that mirrors the classical architecture but uses a variational circuit."""

    class QLayer(tq.QuantumModule):
        """Parameterized quantum layer with random and trainable gates."""

        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(
                n_ops=60, wires=list(range(self.n_wires))
            )
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.crx = tq.CRX(has_params=True, trainable=True)
            self.cry = tq.CRY(has_params=True, trainable=True)
            self.rzz = tq.RZZ(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            self.rx(qdev, wires=0)
            self.ry(qdev, wires=1)
            self.rz(qdev, wires=2)
            self.crx(qdev, wires=[0, 3])
            self.cry(qdev, wires=[1, 2])
            self.rzz(qdev, wires=[0, 1])
            tqf.hadamard(
                qdev,
                wires=3,
                static=self.static_mode,
                parent_graph=self.graph,
            )
            tqf.sx(
                qdev,
                wires=2,
                static=self.static_mode,
                parent_graph=self.graph,
            )
            tqf.cnot(
                qdev,
                wires=[3, 0],
                static=self.static_mode,
                parent_graph=self.graph,
            )

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict["4x4_ryzxy"]
        )
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.binary_head = nn.Linear(self.n_wires, 1)
        self.multi_head = nn.Linear(self.n_wires, 4)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.LogSoftmax(dim=1)
        self.batch_norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return tuple (binary_logits, multi_logits) derived from the quantum measurement."""
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires,
            bsz=bsz,
            device=x.device,
            record_op=True,
        )
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        out = self.batch_norm(out)
        binary = self.sigmoid(self.binary_head(out))
        multi = self.softmax(self.multi_head(out))
        return binary, multi

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the quantum‑encoded feature vector before the classification heads."""
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires,
            bsz=bsz,
            device=x.device,
            record_op=True,
        )
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.batch_norm(out)


__all__ = ["QuantumNATEnhanced"]
