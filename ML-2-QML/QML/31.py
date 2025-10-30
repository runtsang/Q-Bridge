"""Variational Quantum circuit with adaptive measurement for Quantum‑NAT."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf


class AdaptiveAnsatz(tq.QuantumModule):
    """Parameterized ansatz with alternating RX/RZ layers and entangling CNOTs."""

    def __init__(self, n_wires: int = 4, depth: int = 3):
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        self.params = nn.ParameterList()
        for _ in range(depth):
            # RX and RZ on each wire
            self.params.append(nn.Parameter(torch.randn(n_wires)))
            # CNOT ladder
            self.params.append(nn.Parameter(torch.randn(n_wires - 1)))

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice) -> None:
        for i in range(self.depth):
            # Apply RX
            tqf.rx(qdev, wires=range(self.n_wires), params=self.params[i * 2], static=self.static_mode,
                   parent_graph=self.graph)
            # Apply RZ
            tqf.rz(qdev, wires=range(self.n_wires), params=self.params[i * 2 + 1], static=self.static_mode,
                   parent_graph=self.graph)
            # Entangling CNOT ladder
            for w in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[w, w + 1], static=self.static_mode, parent_graph=self.graph)


class QuantumNATModel(tq.QuantumModule):
    """Hybrid quantum encoder and adaptive ansatz for image classification."""

    def __init__(self, n_wires: int = 4, num_classes: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.ansatz = AdaptiveAnsatz(n_wires=n_wires, depth=4)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.classifier = nn.Sequential(
            nn.Linear(n_wires, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            nn.BatchNorm1d(num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        # Pre‑processing: average pool and flatten
        pooled = torch.nn.functional.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.ansatz(qdev)
        out = self.measure(qdev)
        # Classical post‑processing
        out = self.classifier(out)
        return out


__all__ = ["QuantumNATModel"]
