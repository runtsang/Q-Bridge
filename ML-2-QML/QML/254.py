"""Hybrid quantum‑classical model with a parameterized ansatz and classical encoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class QuantumNATEnhanced(tq.QuantumModule):
    """Hybrid architecture: classical encoder → variational circuit → measurement."""

    class QLayer(tq.QuantumModule):
        """Parameterized circuit with rotation and entangling layers."""

        def __init__(self, n_wires: int = 4, n_layers: int = 3):
            super().__init__()
            self.n_wires = n_wires
            self.n_layers = n_layers
            # Single‑qubit rotations
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            # Entangling gate
            self.cnot = tq.CNOT

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            for _ in range(self.n_layers):
                for w in range(self.n_wires):
                    self.rx(qdev, wires=w)
                    self.ry(qdev, wires=w)
                    self.rz(qdev, wires=w)
                # Ring entanglement
                for w in range(self.n_wires):
                    self.cnot(qdev, wires=[w, (w + 1) % self.n_wires])

    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        # Encoder maps 2‑D patches to qubit states
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer(n_wires=self.n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True
        )
        # Classical pre‑processing: average‑pool to match encoder size
        pooled = F.avg_pool2d(x, kernel_size=6).view(bsz, -1)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)


__all__ = ["QuantumNATEnhanced"]
