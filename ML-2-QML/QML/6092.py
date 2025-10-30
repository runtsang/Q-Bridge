"""Quantum module that complements the classical extractor in HybridQuantumNAT."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class HybridQuantumNAT(tq.QuantumModule):
    """
    Quantum backend that receives classical features, encodes them onto a 4‑qubit
    register, applies a variational circuit, and measures Pauli‑Z to produce
    a 4‑dimensional feature vector.  The output is batch‑normed and ready to be
    concatenated with the classical representation.
    """

    class QCLayer(tq.QuantumModule):
        """
        Parameterised quantum circuit that acts as a fully connected layer.
        It consists of a random layer followed by trainable single‑qubit rotations
        and a small entangling block.
        """

        def __init__(self, n_wires: int = 4):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(
                n_ops=75,
                wires=list(range(self.n_wires)),
                seed=42,
            )
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.crx = tq.CRX(has_params=True, trainable=True)
            self.measure = tq.MeasureAll(tq.PauliZ)
            self.norm = nn.BatchNorm1d(self.n_wires)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            self.rx(qdev, wires=0)
            self.ry(qdev, wires=1)
            self.rz(qdev, wires=3)
            self.crx(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict["4x4_ryzxy"]
        )
        self.q_layer = self.QCLayer(self.n_wires)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes a 2‑D image tensor into a quantum state, runs the variational block
        and returns a batch‑normalised 4‑dimensional feature vector.
        """
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
        out = self.q_layer.measure(qdev)
        return self.norm(out)


__all__ = ["HybridQuantumNAT"]
