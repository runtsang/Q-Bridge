"""Hybrid quantum classifier using torchquantum."""
from __future__ import annotations

from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class HybridQuantumClassifier(tq.QuantumModule):
    """Quantum‑enhanced classifier that couples a CNN feature extractor with a variational circuit."""

    class QLayer(tq.QuantumModule):
        """Variational block with randomized and trainable gates."""

        def __init__(self, n_wires: int, depth: int):
            super().__init__()
            self.n_wires = n_wires
            self.depth = depth
            # Randomized circuit for feature injection
            self.random_layer = tq.RandomLayer(
                n_ops=50, wires=list(range(n_wires))
            )
            # Trainable single‑qubit rotations
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            # Two‑qubit entanglers
            self.crx = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)
                self.rz(qdev, wires=w)
            for w in range(self.n_wires - 1):
                self.crx(qdev, wires=[w, w + 1])

    def __init__(
        self,
        num_qubits: int = 4,
        depth: int = 2,
        encoder_type: str = "4x4_ryzxy",
    ) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.depth = depth
        # Encoder that maps classical features to quantum gates
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[encoder_type]
        )
        self.qlayer = self.QLayer(num_qubits, depth)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(num_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        # Extract features with the same CNN as in the classical branch
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        qdev = tq.QuantumDevice(
            n_wires=self.num_qubits, bsz=bsz, device=x.device, record_op=True
        )
        # Encode classical features into quantum gates
        self.encoder(qdev, pooled)
        # Variational layer
        self.qlayer(qdev)
        # Measurement and batch normalisation
        out = self.measure(qdev)
        return self.norm(out)


__all__ = ["HybridQuantumClassifier"]
