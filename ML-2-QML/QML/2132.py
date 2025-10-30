import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumNATEnhanced(tq.QuantumModule):
    """Quantum‑inspired hybrid model retaining the core QFCModel pipeline but
    adding a learnable classical encoder and a contrastive projection head."""
    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=80, wires=list(range(self.n_wires)))
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
            tqf.hadamard(qdev, wires=3, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], parent_graph=self.graph)

    def __init__(self, num_classes: int = 4, proj_dim: int = 64):
        super().__init__()
        self.n_wires = 4
        # Classical encoder: 1×1 conv to map input to 4‑dimensional feature vector
        self.encoder = nn.Conv2d(1, 4, kernel_size=1, stride=1, padding=0)
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.classifier = nn.Linear(self.n_wires, num_classes)
        self.projection = nn.Linear(self.n_wires, proj_dim)
        self.norm_cls = nn.BatchNorm1d(num_classes)
        self.norm_proj = nn.BatchNorm1d(proj_dim)

    def forward(self, x: torch.Tensor):
        bsz = x.shape[0]
        # 1. Classical feature extraction
        feat = self.encoder(x).view(bsz, -1)
        # 2. Quantum device allocation
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz,
                                device=x.device, record_op=True)
        # 3. Basis encoding (Hadamard on all wires)
        tqf.hadamard(qdev, wires=list(range(self.n_wires)), parent_graph=self.graph)
        # 4. Variational circuit
        self.q_layer(qdev)
        # 5. Measurement
        out = self.measure(qdev)
        # 6. Classification logits
        logits = self.classifier(out)
        logits = self.norm_cls(logits)
        # 7. Contrastive projection
        proj = self.projection(out)
        proj = self.norm_proj(proj)
        return logits, proj

__all__ = ["QuantumNATEnhanced"]
