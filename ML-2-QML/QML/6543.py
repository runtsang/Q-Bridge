"""Hybrid quantum‑classical model that couples a CNN feature extractor
with a variational circuit.

The classical CNN is identical to the one used in the classical
HybridNATModel.  The output of the CNN is encoded into a 4‑wire
quantum device using a general Ry‑rotation encoder.  A random layer
followed by trainable RX/RY gates emulates a shallow variational
circuit.  The measurement of all qubits is passed through a linear
head to produce four logits (classification) or a single scalar
(regression).  The implementation keeps the forward pass fully
differentiable by leveraging torchquantum’s static graph support.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class HybridQuantumNATModel(tq.QuantumModule):
    """Hybrid quantum‑classical model for the Quantum‑NAT benchmark.

    Parameters
    ----------
    task : str
        ``'classification'`` or ``'regression'``.  The output head is
        adjusted accordingly.
    """

    class QLayer(tq.QuantumModule):
        """Shallow variational block with random and trainable gates."""
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, task: str = "classification"):
        super().__init__()
        self.task = task
        self.n_wires = 4

        # Classical CNN backbone
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Quantum encoder
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{self.n_wires}xRy"]
        )
        self.q_layer = self.QLayer(self.n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Output head
        if self.task == "classification":
            self.head = nn.Linear(self.n_wires, 4)
        else:  # regression
            self.head = nn.Linear(self.n_wires, 1)

        self.norm = nn.BatchNorm1d(4 if self.task == "classification" else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        # Classical feature extraction
        feat = self.features(x)
        pooled = F.avg_pool2d(feat, 6).view(bsz, 16)  # 16 channels

        # Quantum device
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True
        )
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        qfeat = self.measure(qdev)

        out = self.head(qfeat)
        return self.norm(out.squeeze(-1))

__all__ = ["HybridQuantumNATModel"]
