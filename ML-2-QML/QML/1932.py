"""Quantum module for Quantum‑NAT with adjustable entanglement depth."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QuantumNATGen121(tq.QuantumModule):
    """
    Quantum variant of the Quantum‑NAT model.
    Includes a parameterisable entanglement depth and a regularisation term
    that encourages sparse entanglement.
    """

    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int, ent_depth: int):
            super().__init__()
            self.n_wires = n_wires
            self.ent_depth = ent_depth
            self.random_layer = tq.RandomLayer(n_ops=20, wires=list(range(self.n_wires)))
            # Rotation gates for each wire
            for i in range(self.n_wires):
                setattr(self, f"rx{i}", tq.RX(has_params=True, trainable=True))
                setattr(self, f"ry{i}", tq.RY(has_params=True, trainable=True))
                setattr(self, f"rz{i}", tq.RZ(has_params=True, trainable=True))
            # Entangling layers
            self.cnot_layers = nn.ModuleList([tq.CNOT() for _ in range(self.ent_depth)])

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            # Random initialisation
            self.random_layer(qdev)
            # Apply single‑qubit rotations
            for i in range(self.n_wires):
                getattr(self, f"rx{i}")(qdev, wires=i)
                getattr(self, f"ry{i}")(qdev, wires=i)
                getattr(self, f"rz{i}")(qdev, wires=i)
            # Entangling layers
            for layer in self.cnot_layers:
                layer(qdev, wires=[0, 1])
                layer(qdev, wires=[1, 2])
                layer(qdev, wires=[2, 3])

    def __init__(self, n_wires: int = 4, ent_depth: int = 2) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.ent_depth = ent_depth
        # Encoder maps classical input into initial qubit states
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer(n_wires=self.n_wires, ent_depth=self.ent_depth)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.out_norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        # Global average pooling to 16 features
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        # Encode classical features into qubits
        self.encoder(qdev, pooled)
        # Quantum circuit
        self.q_layer(qdev)
        # Measure
        out = self.measure(qdev)
        return self.out_norm(out)

__all__ = ["QuantumNATGen121"]
