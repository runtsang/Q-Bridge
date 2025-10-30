"""Quantum variant of the Quantum‑NAT hybrid model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class QuantumNATHybrid(tq.QuantumModule):
    """Quantum variant with a parametric encoder and a lightweight variational circuit."""

    class QEncoder(tq.QuantumModule):
        """Encode classical features into a quantum state using a parametric RY gate per wire."""

        def __init__(self, n_wires: int, feature_dim: int):
            super().__init__()
            self.n_wires = n_wires
            self.linear = nn.Linear(feature_dim, n_wires)

        def forward(self, qdev: tq.QuantumDevice, features: torch.Tensor):
            params = self.linear(features)
            for i in range(self.n_wires):
                tq.RY(params[:, i], qdev, wires=i, has_params=True)
            # Optional entanglement
            if self.n_wires > 1:
                tq.CNOT(qdev, wires=[0, 1], has_params=False)

    class VariationalLayer(tq.QuantumModule):
        """Parameter‑efficient variational layer with alternating RZ/RX and CX gates."""

        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.rx = tq.RX(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.cx = tq.CNOT(has_params=False)

        def forward(self, qdev: tq.QuantumDevice):
            for i in range(self.n_wires):
                self.rx(qdev, wires=i)
                self.rz(qdev, wires=i)
                next_wire = (i + 1) % self.n_wires
                self.cx(qdev, wires=[i, next_wire])

    def __init__(self, in_channels: int = 1, num_classes: int = 4):
        super().__init__()
        self.n_wires = 4
        self.encoder = self.QEncoder(self.n_wires, feature_dim=16)
        self.var_layer = self.VariationalLayer(self.n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.size(0)
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires,
            bsz=bsz,
            device=x.device,
            record_op=True,
        )
        # Feature extraction: average pooling to 16 features
        pooled = F.avg_pool2d(x, kernel_size=6).view(bsz, -1)
        # Encode into quantum state
        self.encoder(qdev, pooled)
        # Variational layer
        self.var_layer(qdev)
        # Measurement
        out = self.measure(qdev)
        return self.norm(out)


__all__ = ["QuantumNATHybrid"]
