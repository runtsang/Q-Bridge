import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumNATEnhanced(tq.QuantumModule):
    """Hybrid classical‑quantum encoder with dual‑output head.

    Extends the original QFCModel:
    1. Classical CNN encoder with two convolutional layers.
    2. Variational block with 80 random ops and a deeper entangling pattern.
    3. Two output heads – classification (4 classes) and a 2‑dim vector.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            # Larger random layer for richer feature space
            self.random_layer = tq.RandomLayer(n_ops=80, wires=list(range(self.n_wires)))
            # Parameterised rotations
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)
            self.cnot = tq.CNOT()

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            # Entangling pattern
            self.cnot(qdev, wires=[0, 1])
            self.cnot(qdev, wires=[2, 3])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.batch_norm = nn.BatchNorm1d(self.n_wires)

        # Dual heads
        self.class_head = nn.Sequential(
            nn.Linear(self.n_wires, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 4)
        )
        self.recon_head = nn.Linear(self.n_wires, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        out = self.batch_norm(out)
        cls = self.class_head(out)
        recon = self.recon_head(out)
        return cls, recon
