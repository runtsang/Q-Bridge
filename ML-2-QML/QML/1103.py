"""Quantum variant of QFCModel using a parameterized variational circuit and classical post‑processing."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class VariationalAnsatz(tq.QuantumModule):
    """Parameterized circuit with alternating single‑qubit rotations and entangling CNOTs."""
    def __init__(self, n_wires: int, n_layers: int = 3):
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.rotation = tq.RotationXYZ(has_params=True, trainable=True)
        self.cnot = tq.CNOT(has_params=False, trainable=False)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice):
        for _ in range(self.n_layers):
            for w in range(self.n_wires):
                self.rotation(qdev, wires=w)
            # Entangle adjacent qubits in a ring
            for w in range(self.n_wires):
                self.cnot(qdev, wires=[w, (w + 1) % self.n_wires])
        # Final layer of rotations
        for w in range(self.n_wires):
            self.rotation(qdev, wires=w)


class QFCModel(tq.QuantumModule):
    """Hybrid quantum model with variational ansatz and classical readout."""

    def __init__(self, n_wires: int = 4, n_ansatz_layers: int = 3):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.ansatz = VariationalAnsatz(n_wires=self.n_wires, n_layers=n_ansatz_layers)
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Classical post‑processing MLP
        self.post = nn.Sequential(
            nn.Linear(self.n_wires, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 4)
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)

        # Encode classical features into qubit states
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)

        # Variational layer
        self.ansatz(qdev)

        # Measurement
        out = self.measure(qdev)
        # Classical post‑processing
        out = self.post(out)
        return self.norm(out)


__all__ = ["QFCModel"]
