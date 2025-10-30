"""Hybrid natural image classifier – quantum implementation.

The architecture mirrors the classical version but replaces the ConvFilter
with a 4‑qubit quantum circuit.  Image patches are encoded using a
general encoder, processed by a parameterised random layer plus local
rotations, and the measurement outcome is fed into a classical
fully‑connected head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from torch import Tensor


class HybridQuantumLayer(tq.QuantumModule):
    """Parameterised layer that applies a random circuit followed by
    local rotations and a controlled‑RX.  It is analogous to the
    QLayer in the original seed but with a slightly different
    topology to improve expressivity.
    """
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.random_layer = tq.RandomLayer(n_ops=60, wires=list(range(self.n_wires)))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)
        self.crx = tq.CRX(has_params=True, trainable=True)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice) -> None:
        self.random_layer(qdev)
        self.rx(qdev, wires=0)
        self.ry(qdev, wires=1)
        self.rz(qdev, wires=2)
        self.crx(qdev, wires=[0, 3])
        tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
        tqf.cnot(qdev, wires=[0, 2], static=self.static_mode, parent_graph=self.graph)


class HybridNATModel(tq.QuantumModule):
    """Quantum hybrid model that encodes image patches into a 4‑qubit device
    and measures all qubits.  The measurement vector is normalised and
    passed through a classical fully‑connected head.
    """
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        # Use a 4‑qubit encoder that applies a 4×4 Ry‑Rz‑XY‑Ry rotation
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.quantum_layer = HybridQuantumLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Classical head mirroring the classical version
        self.fc = nn.Sequential(
            nn.Linear(self.n_wires + 1, 64),  # +1 for the ConvFilter scalar
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: Tensor) -> Tensor:
        bsz = x.shape[0]
        # Encode the entire image into a 4-qubit device
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        # Average‑pool to produce a single scalar patch (mimicking ConvFilter)
        pooled = F.avg_pool2d(x, kernel_size=6).view(bsz, 16)  # (B, 16)
        self.encoder(qdev, pooled)
        self.quantum_layer(qdev)
        out = self.measure(qdev)  # (B, 4)
        # Classical ConvFilter scalar (average pooling)
        qfeat = F.avg_pool2d(x, kernel_size=6).view(bsz, 1)   # (B, 1)
        concat = torch.cat([out, qfeat], dim=1)                # (B, 5)
        logits = self.fc(concat)
        return self.norm(logits)


__all__ = ["HybridNATModel"]
