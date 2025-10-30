"""Quantum‑classical hybrid model for Quantum‑NAT with parameter‑shared rotation layers."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class QFCModel(tq.QuantumModule):
    """Hybrid quantum–classical model that extends the original seed by sharing rotation
    parameters across multiple layers and adding a classical readout projection.
    """

    class QLayer(tq.QuantumModule):
        """Parameter‑shared rotation block repeated over several depth layers."""
        def __init__(self, n_wires: int, depth: int = 2) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.depth = depth
            self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(n_wires)))
            # Rotation gates with shared parameters
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.crx = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            for _ in range(self.depth):
                self.random_layer(qdev)
                self.rx(qdev, wires=0)
                self.ry(qdev, wires=1)
                self.rz(qdev, wires=2)
                self.crx(qdev, wires=[0, 3])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self, *, n_wires: int = 4, depth: int = 2) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer(n_wires, depth)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.classical_readout = nn.Linear(n_wires, 4)
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Input image tensor of shape [batch, 1, 28, 28].

        Returns
        -------
        torch.Tensor
            Normalized 4‑dimensional feature vector per sample.
        """
        batch_size = x.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=batch_size, device=x.device, record_op=True
        )
        pooled = F.avg_pool2d(x, 6).view(batch_size, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)               # [B, n_wires]
        out = self.classical_readout(out)      # [B, 4]
        return self.norm(out)


__all__ = ["QFCModel"]
