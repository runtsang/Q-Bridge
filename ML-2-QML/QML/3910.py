"""Hybrid NAT model – quantum implementation.

The quantum branch encodes image features into a qubit register, applies
a random‑layer variational circuit, measures all wires, and feeds the
outcome into a classical head.  It inherits from ``tq.QuantumModule``
for seamless integration with TorchQuantum.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

__all__ = ["HybridNATModel"]


class HybridNATModel(tq.QuantumModule):
    """Quantum‑enhanced NAT model.

    Parameters
    ----------
    in_channels : int, default 1
        Number of input channels.
    output_dim : int, default 4
        Output dimensionality.  Use ``1`` for regression.
    n_wires : int, default 4
        Size of the quantum register.
    """

    class QLayer(tq.QuantumModule):
        """Variational circuit with a random layer and trainable rotations."""

        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)
                self.rz(qdev, wires=wire)

    def __init__(self, in_channels: int = 1, output_dim: int = 4,
                 n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Classical encoder identical to the CNN backbone's initial layers
        # but used only to generate a feature map for the quantum state.
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.q_layer = self.QLayer(n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(n_wires, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        bsz = x.shape[0]
        # Classical feature extraction
        features = self.encoder(x).view(bsz, -1)[:, :self.n_wires]
        # Prepare quantum device
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz,
                                device=x.device, record_op=False)
        # Encode the classical features into the quantum state
        self.encoder(qdev, features)
        # Variational circuit
        self.q_layer(qdev)
        # Measurement
        out = self.measure(qdev)
        # Classical head
        out = self.head(out)
        return self.bn(out)
