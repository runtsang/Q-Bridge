"""Quantum‑aware variant of QuantumNATEnhanced using torchquantum."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class QLayer(tq.QuantumModule):
    """Parameterized variational circuit for the bottleneck."""
    def __init__(self, n_wires: int = 8, n_layers: int = 3):
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        # Parameters for RX, RY, RZ on each wire per layer
        self.params = nn.Parameter(torch.randn(n_layers, n_wires, 3))
        self.cnot = tq.CNOT
        self.rx = tq.RX
        self.ry = tq.RY
        self.rz = tq.RZ

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice):
        for layer in range(self.n_layers):
            for wire in range(self.n_wires):
                # Apply single‑qubit rotations
                self.rx(qdev, wires=wire, params=self.params[layer, wire, 0])
                self.ry(qdev, wires=wire, params=self.params[layer, wire, 1])
                self.rz(qdev, wires=wire, params=self.params[layer, wire, 2])
            # Entangling layer
            for wire in range(self.n_wires - 1):
                self.cnot(qdev, wires=[wire, wire + 1])


class QuantumNATEnhanced(tq.QuantumModule):
    """Hybrid model that encodes classical features into a quantum state,
    processes them with a variational circuit, and measures the result."""
    def __init__(self, num_classes: int = 4, n_wires: int = 8, n_layers: int = 3):
        super().__init__()
        self.n_wires = n_wires
        # Encoder that maps pooled features to rotations
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = QLayer(n_wires=n_wires, n_layers=n_layers)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        # Global average pool to match feature dimension
        pooled = F.avg_pool2d(x, 6).view(bsz, -1)
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)


__all__ = ["QuantumNATEnhanced"]
