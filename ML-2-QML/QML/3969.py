"""Quantum‑centric hybrid model featuring a variational circuit and a random quantum filter.

`QuantumNATHybrid` extends `torchquantum.QuantumModule` and embeds a classical
feature encoder, a lightweight quantum filter (random layer + fixed gates), and
a variational block that mirrors the structure from the original Quantum‑NAT
implementation.  The model accepts a batch of single‑channel images, averages
them to a 16‑dimensional vector, encodes this vector into a 4‑wire quantum
device, applies the filter and variational layers, and finally measures all
qubits in the Pauli‑Z basis.  A batch‑normalisation layer stabilises the
output.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumNATHybrid(tq.QuantumModule):
    """Quantum‑centric hybrid model featuring a variational circuit and a random quantum filter."""
    class QFilter(tq.QuantumModule):
        """Random quantum filter that emulates the behaviour of a classical Conv filter."""
        def __init__(self, n_wires: int = 4) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            tqf.hadamard(qdev, wires=range(self.n_wires))
            tqf.cnot(qdev, wires=[0, 1])
            tqf.cnot(qdev, wires=[2, 3])

    class QLayer(tq.QuantumModule):
        """Variational block inspired by the original Quantum‑NAT q‑layer."""
        def __init__(self) -> None:
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

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_filter = self.QFilter(n_wires=self.n_wires)
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        # Feature extraction: average pool to 16‑dimensional vector
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        # Encode classical features into quantum state
        self.encoder(qdev, pooled)
        # Quantum filter
        self.q_filter(qdev)
        # Variational layer
        self.q_layer(qdev)
        # Measure all qubits
        out = self.measure(qdev)
        return self.norm(out)

__all__ = ["QuantumNATHybrid"]
