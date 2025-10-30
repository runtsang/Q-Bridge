"""Quantum variational model for Quantum‑NAT with parameterized rotations and entanglement."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class QuantumNATModel(tq.QuantumModule):
    """Variational quantum circuit with encoder and entangling layers for Quantum‑NAT."""

    class Encoder(tq.QuantumModule):
        """Parameter‑free encoding using Ry and Rz rotations derived from image patches."""

        def __init__(self, n_wires: int = 4):
            super().__init__()
            self.n_wires = n_wires

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor) -> None:
            # Map each pixel to a rotation angle
            for wire in range(self.n_wires):
                theta = x[:, wire]  # shape (bsz,)
                tqf.ry(qdev, theta, wires=wire, static=self.static_mode, parent_graph=self.graph)
                tqf.rz(qdev, theta, wires=wire, static=self.static_mode, parent_graph=self.graph)

    class VariationalLayer(tq.QuantumModule):
        """Parameterized rotation and entangling layer."""

        def __init__(self, n_wires: int = 4):
            super().__init__()
            self.n_wires = n_wires
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.cnot = tq.CNOT

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)
                self.rz(qdev, wires=wire)
            # Entangle neighboring qubits in a ring
            for wire in range(self.n_wires):
                self.cnot(qdev, wires=[wire, (wire + 1) % self.n_wires])

    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = self.Encoder(n_wires)
        self.var_layer = self.VariationalLayer(n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        # Prepare device
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires,
            bsz=bsz,
            device=x.device,
            record_op=True,
        )
        # Encode input
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        encoded = pooled[:, : self.n_wires]
        self.encoder(qdev, encoded)
        # Variational layer
        self.var_layer(qdev)
        # Measurement
        out = self.measure(qdev)
        return self.norm(out)


__all__ = ["QuantumNATModel"]
