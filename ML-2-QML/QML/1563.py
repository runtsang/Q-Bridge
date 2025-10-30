"""Quantum variant of QFCModelEnhanced with a variational circuit and density‑matrix readout."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Tuple

class QFCModelEnhanced(tq.QuantumModule):
    """Quantum fully connected model with variational layer and density‑matrix output.
    Returns logits and predictive entropy derived from measurement statistics."""
    class VariationalLayer(tq.QuantumModule):
        def __init__(self, n_wires: int = 4):
            super().__init__()
            self.n_wires = n_wires
            # Parameter‑shiftable rotation layer with learnable angles
            self.param_layer = tq.RY(has_params=True, trainable=True)
            self.cnot = tq.CNOT
            # Parameter‑shiftable entangling block
            self.entangle = tq.CRX(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice) -> None:
            # Apply RY rotations to each wire
            for w in range(self.n_wires):
                self.param_layer(qdev, wires=w)
            # Entangle wires in a ring topology
            for i in range(self.n_wires):
                self.entangle(qdev, wires=[i, (i + 1) % self.n_wires])

    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        # Encoder: use a 4‑qubit general encoder with RyZXY pattern
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.variational = self.VariationalLayer(n_wires=self.n_wires)
        # Readout: density matrix on all qubits
        self.dm_measure = tq.MeasureDensityMatrix()
        # Classical linear head mapping flattened density matrix to logits
        self.linear = nn.Linear(2 ** self.n_wires, 10)
        self.norm = nn.BatchNorm1d(10)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode, evolve, measure, and return logits + entropy.
        """
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz,
                                device=x.device, record_op=True)
        # Pool image and flatten to 16‑dim vector
        pooled = F.avg_pool2d(x, kernel_size=6).view(bsz, 16)
        # Encoder transforms classical data into circuit state
        self.encoder(qdev, pooled)
        # Variational circuit
        self.variational(qdev)
        # Density‑matrix readout; shape (bsz, 2**n, 2**n)
        dm = self.dm_measure(qdev)
        # Flatten each density matrix into a vector of real numbers
        dm_vec = dm.reshape(bsz, -1).float()
        logits = self.linear(dm_vec)
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1, keepdim=True)
        return self.norm(logits), entropy

__all__ = ["QFCModelEnhanced"]
