"""QuantumNATEnhanced: quantum model with hybrid variational layer and residual mixing."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumNATEnhanced(tq.QuantumModule):
    """Quantum model extending the seed with a hybrid variational layer and residual mixing."""
    class QLayer(tq.QuantumModule):
        """Variational layer that entangles 4 qubits with learnable parameters."""
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            # Randomized circuit followed by trainable rotations
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.crx = tq.CRX(has_params=True, trainable=True)
            self.crz = tq.CRZ(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            # Random layer
            self.random_layer(qdev)
            # Parameterized rotations on each qubit
            self.rx(qdev, wires=0)
            self.ry(qdev, wires=1)
            self.rz(qdev, wires=2)
            # Entangling gates
            self.crx(qdev, wires=[0, 3])
            self.crz(qdev, wires=[1, 2])
            # Additional CNOTs to increase entanglement
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[2, 1], static=self.static_mode, parent_graph=self.graph)
            # Hadamards to create superposition
            tqf.hadamard(qdev, wires=0, static=self.static_mode, parent_graph=self.graph)
            tqf.hadamard(qdev, wires=1, static=self.static_mode, parent_graph=self.graph)

    def __init__(self, num_classes: int = 4, hidden_dim: int = 64) -> None:
        super().__init__()
        self.n_wires = 4
        # Classical encoder
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        # Hybrid variational layer
        self.q_layer = self.QLayer()
        # Measurement
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Classifier to combine quantum and classical outputs
        self.class_proj = nn.Linear(16, self.n_wires)
        self.classifier = nn.Linear(self.n_wires, num_classes)
        # Batch norm
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        # Quantum device
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        # Classical feature extraction: average pooling to match encoder input size
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        # Hybrid variational layer
        self.q_layer(qdev)
        # Quantum measurement
        q_out = self.measure(qdev)          # shape: (bsz, 4)
        # Residual: combine with classical features
        flat = pooled.view(bsz, -1)         # shape: (bsz, 16)
        class_feat = self.class_proj(flat)  # shape: (bsz, 4)
        combined = q_out + class_feat
        # Classification
        logits = self.classifier(combined)
        return self.norm(logits)

__all__ = ["QuantumNATEnhanced"]
