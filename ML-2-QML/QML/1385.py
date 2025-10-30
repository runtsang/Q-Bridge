"""Quantum variant of QFCModel with a 6‑qubit variational circuit and classical post‑processing.

Features
--------
- 6‑wire quantum device for richer expressivity.
- Two variational layers: a random layer followed by a parameterized entangling block.
- Classical encoder that maps pooled image features to qubit rotations.
- Measurement of Pauli‑Z expectation values.
- A linear classifier mapping quantum outputs to 4 logits.
- Supports static graph execution for efficient back‑propagation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class QFCModel(tq.QuantumModule):
    """
    Quantum fully connected model with a 6‑qubit variational circuit.

    Parameters
    ----------
    n_wires : int
        Number of qubits. Default is 6.
    num_classes : int
        Number of output classes. Default is 4.
    """

    class VariationalLayer(tq.QuantumModule):
        """Parameterized entangling block using RY, RZ, and CRX gates."""

        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=80, wires=list(range(n_wires)))
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.crx = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            for w in range(self.n_wires):
                self.ry(qdev, wires=w)
                self.rz(qdev, wires=w)
            # Entangle neighboring qubits
            for w in range(self.n_wires - 1):
                self.crx(qdev, wires=[w, w + 1])

    def __init__(
        self,
        n_wires: int = 6,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict["6x6_ryzxy"]
        )
        self.var_layer = self.VariationalLayer(n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.classifier = nn.Linear(n_wires, num_classes)
        self.out_norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True
        )
        # Classical pooling to match encoder input size
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        # Encode classical features into qubit rotations
        self.encoder(qdev, pooled)
        # Variational circuit
        self.var_layer(qdev)
        # Measure expectation values
        out = self.measure(qdev)
        # Classical post‑processing
        out = self.classifier(out)
        return self.out_norm(out)

    def get_quantum_state(self, x: torch.Tensor) -> torch.Tensor:
        """Return the raw statevector for analysis."""
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=False
        )
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.var_layer(qdev)
        return qdev.state

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_wires={self.n_wires}, num_classes={self.out_norm.num_features})"


__all__ = ["QFCModel"]
