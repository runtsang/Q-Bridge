from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class HybridQCNNModel(nn.Module):
    """Hybrid classical‑quantum CNN inspired by QCNN and Quantum‑NAT."""
    def __init__(self) -> None:
        super().__init__()
        # Classical feature extractor (Quantum‑NAT style)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Quantum feature extractor (QCNN style)
        self.q_layer = self.QLayer()
        # Post‑quantum classifier
        self.fc = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.norm = nn.BatchNorm1d(4)

    class QLayer(tq.QuantumModule):
        """Quantum block that mimics a QCNN convolution‑pooling layer."""
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(self.n_wires)))
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.crx = tq.CRX(has_params=True, trainable=True)
            self.measure = tq.MeasureAll(tq.PauliZ)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            # Randomised entangling block
            self.random_layer(qdev)
            # Parameterised rotations
            self.rz(qdev, wires=0)
            self.ry(qdev, wires=1)
            self.crx(qdev, wires=[0, 2])
            # Mixing operation
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        # Classical feature extraction
        features = self.features(x)
        pooled = F.avg_pool2d(features, 6).view(bsz, -1)
        # Quantum device
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)
        self.q_layer(qdev)
        # Measure all qubits
        out = self.q_layer.measure(qdev)
        out = self.norm(out)
        # Final classifier head
        return self.fc(out)

def QCNN() -> HybridQCNNModel:
    """Factory returning the configured :class:`HybridQCNNModel`."""
    return HybridQCNNModel()

__all__ = ["QCNN", "HybridQCNNModel"]
