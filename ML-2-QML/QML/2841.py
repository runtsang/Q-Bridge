"""Hybrid quantum model combining a classical CNN encoder, a photonic‑style
parameterized quantum layer, and a linear post‑processing head.

The architecture merges the QFCModel quantum module with the
FraudDetection photonic construction: the classical encoder produces a
feature vector that is encoded into a 4‑wire quantum device.  The quantum
circuit uses a RandomLayer followed by rotation gates and a
photonic‑style sequence (Hadamard, SX, CNOT).  After measurement the
quantum output is concatenated with a classical linear projection of the
same encoded vector, producing a richer feature set before batch
normalisation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumNATHybrid(tq.QuantumModule):
    """Hybrid quantum‑classical model merging QFCModel and FraudDetection ideas."""
    class QLayer(tq.QuantumModule):
        """Parameterised quantum layer with RandomLayer and rotation gates."""
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.crx = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            self.rx(qdev, wires=0)
            self.ry(qdev, wires=1)
            self.rz(qdev, wires=3)
            self.crx(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self, n_wires: int = 4, out_dim: int = 4, dropout: float = 0.0):
        super().__init__()
        self.n_wires = n_wires
        # Classical encoder: a shallow CNN that produces a 16‑dim vector
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Classical linear head on the encoded vector
        self.linear_head = nn.Linear(16, 8)
        self.classical_head = nn.Linear(8, out_dim)
        self.norm = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        # Encode input: global average pooling to 16‑dim vector
        pooled = F.avg_pool2d(x, kernel_size=6).view(bsz, 16)
        # Classical linear projection
        cls_feat = self.linear_head(pooled)
        # Quantum encoding
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        q_out = self.measure(qdev)
        # Concatenate quantum and classical features
        out = torch.cat([q_out, cls_feat], dim=1)
        out = self.norm(out)
        return self.dropout(out)

__all__ = ["QuantumNATHybrid"]
