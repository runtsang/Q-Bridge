from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QLayer(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
        self.rx0 = tq.RX(has_params=True, trainable=True)
        self.ry0 = tq.RY(has_params=True, trainable=True)
        self.rz0 = tq.RZ(has_params=True, trainable=True)
        self.crx0 = tq.CRX(has_params=True, trainable=True)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice):
        self.random_layer(qdev)
        self.rx0(qdev, wires=0)
        self.ry0(qdev, wires=1)
        self.rz0(qdev, wires=3)
        self.crx0(qdev, wires=[0, 2])
        tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
        tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
        tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

class QuantumHybridHead(tq.QuantumModule):
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        # Pad the scalar to 4‑dimensional input for the encoder
        padded = torch.cat([x, torch.zeros(bsz, 3, device=x.device)], dim=1)
        self.encoder(qdev, padded)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

class HybridBinaryClassifier(nn.Module):
    """
    Hybrid CNN + quantum head for binary classification.
    The head is implemented with torchquantum and outputs a probability vector.
    """
    def __init__(self) -> None:
        super().__init__()
        # Feature extractor – same topology as in the classical version
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Dropout2d(p=0.5),
        )
        # Fully‑connected projection
        self.fc = nn.Sequential(
            nn.Linear(55815, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 1),
        )
        # Quantum head
        self.quantum_head = QuantumHybridHead(n_wires=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # Pass scalar through quantum hybrid head
        qout = self.quantum_head(x)
        # Convert expectation to probability via sigmoid
        probs = torch.sigmoid(qout)
        return torch.cat([probs, 1 - probs], dim=-1)

__all__ = ["HybridBinaryClassifier"]
