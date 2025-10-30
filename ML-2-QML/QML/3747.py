"""Hybrid quantum model combining encoder, attention‑style variational layer, and measurement."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QAttentionLayer(tq.QuantumModule):
    """Variational layer that implements a learnable self‑attention‑style circuit."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        # Rotation layers on each qubit
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)
        # Controlled rotations between adjacent qubits
        self.crx = tq.CRX(has_params=True, trainable=True)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice):
        # Apply rotations
        for i in range(self.n_wires):
            self.rx(qdev, wires=i)
            self.ry(qdev, wires=i)
            self.rz(qdev, wires=i)
        # Entangling block
        for i in range(self.n_wires - 1):
            self.crx(qdev, wires=[i, i + 1])

class HybridNATModel(tq.QuantumModule):
    """Quantum CNN‑style model with encoder, attention‑style variational layer, and measurement."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.attn_layer = QAttentionLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        # Classical pooling to feed into encoder
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.attn_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

__all__ = ["HybridNATModel"]
