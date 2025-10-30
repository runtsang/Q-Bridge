"""Quantum variant of QFCModel with a variational circuit and classical readout.

Improvements over the original:
1. Uses a 4‑wire variational circuit with alternating layers of RY, RZ, CX, and a
   trainable entangling block.
2. Encodes the pooled classical features into the quantum state via a custom
   embedding circuit.
3. Measures all wires in the Pauli‑Z basis and feeds the expectation values into
   a learnable linear head.
4. Adds a batch‑norm to stabilise the output.

The model can be trained end‑to‑end with gradient descent through the quantum
device.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QFCModel(tq.QuantumModule):
    """Quantum‑NAT model with variational circuit and classical projection."""

    class Encoder(tq.QuantumModule):
        """Encodes a vector of length 16 into a 4‑wire quantum state."""
        def __init__(self, n_wires: int = 4):
            super().__init__()
            self.n_wires = n_wires
            # Use a simple Ry encoding for each wire
            self.ry = tq.RY(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor):
            # x shape: (B, 16)
            # Map 16 features to 4 wires by grouping
            for i in range(self.n_wires):
                idx = slice(i * 4, (i + 1) * 4)
                # Take mean of 4 features for each wire
                wire_val = x[:, idx].mean(dim=1)
                self.ry(qdev, params=wire_val, wires=i)

    class VariationalLayer(tq.QuantumModule):
        """A parameterised entangling layer."""
        def __init__(self, n_wires: int = 4, n_layers: int = 3):
            super().__init__()
            self.n_wires = n_wires
            self.n_layers = n_layers
            # Parameterised rotation layers
            self.rotations = nn.ModuleList(
                [nn.Sequential(
                    tq.RY(has_params=True, trainable=True),
                    tq.RZ(has_params=True, trainable=True),
                ) for _ in range(n_layers)]
            )
            self.cx = tq.CX()

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            for layer in self.rotations:
                layer(qdev, wires=list(range(self.n_wires)))
                # Entangling CZ pattern
                for i in range(self.n_wires):
                    j = (i + 1) % self.n_wires
                    self.cx(qdev, wires=[i, j])

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = self.Encoder(self.n_wires)
        self.var_layer = self.VariationalLayer(self.n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.proj = nn.Linear(self.n_wires, 4)
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        # Classical pooling
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        # Encode
        self.encoder(qdev, pooled)
        # Variational circuit
        self.var_layer(qdev)
        # Measurement
        out = self.measure(qdev)
        # Classical projection
        out = self.proj(out)
        return self.norm(out)

__all__ = ["QFCModel"]
