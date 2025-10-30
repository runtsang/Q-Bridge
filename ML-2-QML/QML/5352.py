"""Hybrid quantum‑classical architecture mirroring the classical version but with real quantum layers.
It uses TorchQuantum to encode data, apply a quanvolution filter, a fully‑connected quantum layer,
measure, and a classical classifier head.  The design demonstrates how each reference pair
contributes to a single, cohesive model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class QuanvolutionFilter(tq.QuantumModule):
    """Apply a random two‑qubit quantum kernel to 2×2 image patches."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice) -> None:
        # Encode input data into qubit states
        self.encoder(qdev, qdev.states)  # placeholder for actual encoding
        self.q_layer(qdev)
        self.measure(qdev)


class QLayer(tq.QuantumModule):
    """Quantum fully‑connected layer inspired by the original QFCModel."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
        self.rx0 = tq.RX(has_params=True, trainable=True)
        self.ry0 = tq.RY(has_params=True, trainable=True)
        self.rz0 = tq.RZ(has_params=True, trainable=True)
        self.crx0 = tq.CRX(has_params=True, trainable=True)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice) -> None:
        self.random_layer(qdev)
        self.rx0(qdev, wires=0)
        self.ry0(qdev, wires=1)
        self.rz0(qdev, wires=3)
        self.crx0(qdev, wires=[0, 2])
        tqf.hadamard(qdev, wires=3)
        tqf.sx(qdev, wires=2)
        tqf.cnot(qdev, wires=[3, 0])


class Kernel(tq.QuantumModule):
    """Quantum kernel evaluated via a fixed TorchQuantum ansatz."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(q_device, x, y)

    def kernel_value(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.forward(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])


class HybridNATModel(tq.QuantumModule):
    """Hybrid quantum‑classical model combining data encoding, a quanvolution filter,
    a quantum fully‑connected layer, measurement, and a classical linear head.
    """
    def __init__(self, n_classes: int = 10, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Data encoder
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        # Quanvolution block
        self.quanv = QuanvolutionFilter()
        # Quantum fully‑connected layer
        self.qfc = QLayer()
        # Measurement and normalization
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)
        # Classical classifier head
        self.classifier = nn.Linear(self.n_wires, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        # Classical preprocessing: average pooling to reduce dimensionality
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        # Quantum device
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        # Encode classical data
        self.encoder(qdev, pooled)
        # Apply quanvolution filter
        self.quanv(qdev)
        # Apply quantum fully‑connected layer
        self.qfc(qdev)
        # Measure all qubits
        out = self.measure(qdev)
        out = self.norm(out)
        # Final classification
        logits = self.classifier(out)
        return logits


__all__ = ["HybridNATModel"]
