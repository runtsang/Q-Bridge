"""QuantumHybridNAT: hybrid model with optional quantum layer.

This module defines the same class name QuantumHybridNAT but now as a
torchquantum.QuantumModule.  It uses the classical backbone from the
classical implementation, then passes the flattened features through a
variational quantum circuit.  The quantum head can be swapped with a
simple sigmoid head by setting `use_quantum=False` during init.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf


class QuantumHybridNAT(tq.QuantumModule):
    """
    Hybrid CNN + quantum layer.  The model is compatible with the
    classical version: if `use_quantum` is False the quantum layer is
    replaced by a sigmoid head.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int = 4):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.cnot = tq.CNOT()
            self.hadamard = tq.Hadamard()
            self.sx = tq.SX()

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            self.rx(qdev, wires=0)
            self.ry(qdev, wires=1)
            self.rz(qdev, wires=2)
            self.cnot(qdev, wires=[0, 2])
            self.hadamard(qdev, wires=1)
            self.sx(qdev, wires=2)
            self.cnot(qdev, wires=[1, 3])

    def __init__(self, in_channels: int = 1, n_filt: int = 8,
                 n_features: int = 64, n_wires: int = 4,
                 use_quantum: bool = True) -> None:
        super().__init__()
        # Classical backbone identical to the ml version
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, n_filt, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(n_filt, n_filt * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.n_flat = n_filt * 2 * 7 * 7
        self.fc = nn.Sequential(
            nn.Linear(self.n_flat, n_features),
            nn.ReLU(inplace=True),
            nn.Linear(n_features, n_wires),
        )
        self.norm = nn.BatchNorm1d(n_wires)
        self.use_quantum = use_quantum
        if self.use_quantum:
            self.q_layer = self.QLayer(n_wires)
            self.measure = tq.MeasureAll(tq.PauliZ)
        else:
            # Classical sigmoid head for quick prototyping
            self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        x = self.features(x)
        x = x.view(bsz, -1)
        x = self.fc(x)
        if self.use_quantum:
            qdev = tq.QuantumDevice(n_wires=self.q_layer.n_wires,
                                    bsz=bsz,
                                    device=x.device,
                                    record_op=True)
            self.q_layer(qdev)
            out = self.measure(qdev)
            return self.norm(out)
        else:
            return self.norm(self.sigmoid(x))


__all__ = ["QuantumHybridNAT"]
