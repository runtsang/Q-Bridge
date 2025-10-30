"""Quantum‑classical hybrid model for Quantum‑NAT.

The model keeps the encoder from the seed but replaces the
variational circuit with a small 2‑layer entanglement network
to reduce parameters.  After measurement we feed the result into
a classical linear layer to obtain a 4‑dimensional output.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class EntangledLayer(tq.QuantumModule):
    """A single entangling block: RX on all wires followed by CRX pairs."""
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.rx = tq.RX(has_params=True, trainable=True)
        self.crx = tq.CRX(has_params=True, trainable=True)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice):
        for w in range(self.n_wires):
            self.rx(qdev, wires=w)
        for w in range(self.n_wires):
            self.crx(qdev, wires=[w, (w + 1) % self.n_wires])

class QuantumNATEnhanced(tq.QuantumModule):
    """Hybrid quantum‑classical model with attention‑inspired encoder."""
    def __init__(self, num_classes: int = 4, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        # Encoder that maps 16‑dim pooled features to qubit states
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        # Two entangling layers
        self.q_layer = nn.Sequential(
            EntangledLayer(n_wires),
            EntangledLayer(n_wires),
        )
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Classical head
        self.classifier = nn.Linear(self.n_wires, num_classes)
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        # Apply the two entangling layers
        for layer in self.q_layer:
            layer(qdev)
        out = self.measure(qdev)
        out = self.classifier(out)
        return self.norm(out)

__all__ = ["QuantumNATEnhanced"]
