"""Hybrid quantum model with classification and regression branches.

The quantum module mirrors the classical architecture: a shared encoder
followed by a parameterized variational layer.  Two measurement‑based
heads provide a 4‑way classification output and a single‑value regression
prediction.  This dual‑head design facilitates direct quantum‑classical
benchmarking of multi‑task learning.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq


class HybridNATModel(tq.QuantumModule):
    """Quantum implementation of the hybrid architecture."""

    class QLayer(tq.QuantumModule):
        """Variational block used by both heads."""

        def __init__(self, n_wires: int, n_ops: int = 30):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)

    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        # Encoder that maps classical features to a quantum state
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{n_wires}xRy"])
        self.q_layer = self.QLayer(n_wires, n_ops=30)
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Heads
        self.class_head = nn.Linear(n_wires, 4)
        self.reg_head = nn.Linear(n_wires, 1)
        self.norm = nn.BatchNorm1d(4)

    def forward_classification(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classification forward pass.
        Input: batch of 1‑channel images.
        Returns batch‑size × 4 logits, batch‑normed.
        """
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires,
            bsz=bsz,
            device=x.device,
            record_op=True,
        )
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(self.class_head(out))

    def forward_regression(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Regression forward pass.
        Input: batch of quantum states (complex tensors).
        Returns batch‑size continuous values.
        """
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires,
            bsz=bsz,
            device=state_batch.device,
        )
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.reg_head(features).squeeze(-1)


__all__ = ["HybridNATModel"]
