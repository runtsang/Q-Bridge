from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from torchquantum import encoder_op_list_name_dict

class QFCQuantumLayer(tq.QuantumModule):
    """Quantum fully‑connected block inspired by the Quantum‑NAT QFCModel."""
    def __init__(self, n_wires: int = 4, n_ops: int = 50):
        super().__init__()
        self.n_wires = n_wires
        self.random_layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(n_wires)))
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

class FraudDetectionHybrid(nn.Module):
    """
    Quantum classifier that accepts a 16‑dim feature vector (output of the classical
    feature extractor) and produces a fraud probability.  The architecture
    mirrors the classical backbone while replacing the final fully‑connected
    layer with a variational quantum circuit.
    """
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.q_layer = QFCQuantumLayer(n_wires=n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(n_wires)
        # Map the quantum measurement to a scalar score
        self.classifier = nn.Linear(n_wires, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: (batch, 16) tensor produced by the classical extractor.
        """
        bsz = features.shape[0]
        # Encode 16‑dim feature vector into a 4‑qubit state
        encoder = tq.GeneralEncoder(encoder_op_list_name_dict["4x4_ryzxy"])
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=features.device, record_op=True)
        encoder(qdev, features)
        self.q_layer(qdev)
        out = self.measure(qdev)
        out = self.norm(out)
        logits = self.classifier(out)
        return logits

__all__ = ["QFCQuantumLayer", "FraudDetectionHybrid"]
