"""Quantum‑NAT enhanced model with a trainable encoder and skip‑connection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumNATEnhanced(tq.QuantumModule):
    """Quantum model with a trainable encoder and residual skip‑connection."""
    class Encoder(tq.QuantumModule):
        """Parameter‑efficient encoder mapping 16‑dimensional input to 4 qubit rotations."""
        def __init__(self, n_wires=4):
            super().__init__()
            self.n_wires = n_wires
            # Linear mapping from 16 features to 4 rotation angles
            self.linear = nn.Linear(16, n_wires)
            # Rotation gates
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor):
            # x shape: (B, 16)
            angles = self.linear(x)  # (B, n_wires)
            for i in range(self.n_wires):
                # Each gate receives a (B, 1) parameter tensor
                theta = angles[:, i].unsqueeze(-1)
                self.rx(qdev, wires=i, params=theta)
                self.ry(qdev, wires=i, params=theta)
                self.rz(qdev, wires=i, params=theta)

    class VariationalLayer(tq.QuantumModule):
        """Variational block with random layer and trainable two‑qubit gates."""
        def __init__(self, n_wires=4):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.crx = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            self.rx(qdev, wires=0)
            self.ry(qdev, wires=1)
            self.rz(qdev, wires=2)
            self.crx(qdev, wires=[0, 3])

    def __init__(self, n_wires=4):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = self.Encoder(n_wires)
        self.q_layer = self.VariationalLayer(n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        # Reduce input to 4x4 feature map
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        # Encode classical features into quantum state
        self.encoder(qdev, pooled)
        # Skip: capture encoded expectation values
        encoded = self.measure(qdev)
        # Variational processing
        self.q_layer(qdev)
        out = self.measure(qdev)
        # Add skip connection
        out = out + encoded
        return self.norm(out)

__all__ = ["QuantumNATEnhanced"]
