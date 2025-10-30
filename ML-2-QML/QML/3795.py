"""Hybrid quantum neural network inspired by Quantum‑NAT.

QHybridModel extends torchquantum.QuantumModule and interleaves
classical pooling with a variational quantum encoder that operates on
the pooled features.  The quantum block uses a RandomLayer followed
by trainable RX/RZ/CRX gates, repeated twice to increase expressive
power.  The model returns four logits after a linear read‑out and
batch‑normalisation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from torchquantum import encoder_op_list_name_dict


class QHybridModel(tq.QuantumModule):
    """Quantum hybrid model with a classical encoder and a variational layer."""

    class QLayer(tq.QuantumModule):
        """Parameterized quantum block with random and trainable gates."""

        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            # Random layer to initialise a rich entangling structure
            self.random = tq.RandomLayer(n_ops=60, wires=list(range(self.n_wires)))
            # Trainable single‑qubit rotations
            self.rx = tq.RX(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.crx = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            # Entangle with a random circuit
            self.random(qdev)
            # Apply a repeated sequence of trainable gates
            for _ in range(2):
                for w in range(self.n_wires):
                    self.rx(qdev, wires=w)
                    self.rz(qdev, wires=w)
                self.crx(qdev, wires=[0, self.n_wires - 1])

    def __init__(self, n_wires: int = 16, num_classes: int = 4):
        super().__init__()
        self.n_wires = n_wires
        # Use a general encoder that maps a real vector into a quantum state
        self.encoder = tq.GeneralEncoder(
            encoder_op_list_name_dict["16x16_ryzxy"]
        )
        self.q_layer = self.QLayer(n_wires=self.n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass that encodes pooled image features into a quantum state."""
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires,
            bsz=bsz,
            device=x.device,
            record_op=True,
        )

        # Classical preprocessing: average pool to 4×4 and flatten to 16 features
        pooled = F.avg_pool2d(x, kernel_size=6, stride=6).view(bsz, -1)
        # Ensure vector length matches number of wires
        if pooled.shape[1]!= self.n_wires:
            raise ValueError(
                f"Expected {self.n_wires} features for encoding, got {pooled.shape[1]}"
            )

        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)


__all__ = ["QHybridModel"]
