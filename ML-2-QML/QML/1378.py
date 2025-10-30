"""Quantum model with a parameter‑efficient ansatz and measurement‑based output.

This version replaces the random layer with a hardware‑efficient ansatz
consisting of alternating RY rotations and CNOTs.  The encoder uses
a 4‑wire 4×4 RyZXY pattern, and the measurement returns the expectation
values of Pauli‑Z on all wires.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class QuantumNATEnhanced(tq.QuantumModule):
    """Quantum model with a parameter‑efficient ansatz and measurement‑based output."""

    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            # Parameter‑efficient ansatz: 3 layers of RY + CNOT
            self.ry = tq.RY(has_params=True, trainable=True)
            self.cnot = tq.CNOT()
            self.layers = 3

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            # Apply layers of RY and CNOT
            for _ in range(self.layers):
                for wire in range(self.n_wires):
                    self.ry(qdev, wires=wire)
                # CNOT ladder
                for i in range(self.n_wires - 1):
                    self.cnot(qdev, wires=[i, i + 1])
            # Add a few single‑qubit rotations for expressivity
            tqf.hadamard(qdev, wires=0)
            tqf.sx(qdev, wires=1)

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True
        )
        # Global average pooling to 16 features
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)


__all__ = ["QuantumNATEnhanced"]
