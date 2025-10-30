"""Quantum‑only model implementing the same functionality as the classical side.

The module is built with TorchQuantum and mimics the hybrid architecture in
full quantum form.  It encodes classical image features into a 4‑qubit
state, runs a highly expressive random variational circuit, measures the
state, and normalises the output with batch‑norm.  The model is fully
differentiable and can be trained end‑to‑end with PyTorch optimisers.

Key components:
* ``Encoder`` – a 4‑qubit Ry‑Rz‑xy encoder (pre‑defined in TorchQuantum).
* ``QLayer`` – 50 random two‑qubit gates plus a small deterministic block
  (RX, RY, RZ, CRX) to increase expressivity.
* ``HybridQuantumNAT`` – the full quantum module that mirrors the classical
  architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf


class QLayer(tq.QuantumModule):
    """Variational layer consisting of random gates and a small deterministic block."""

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
        self.rx0 = tq.RX(has_params=True, trainable=True)
        self.ry0 = tq.RY(has_params=True, trainable=True)
        self.rz0 = tq.RZ(has_params=True, trainable=True)
        self.crx0 = tq.CRX(has_params=True, trainable=True)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice):
        self.random_layer(qdev)
        self.rx0(qdev, wires=0)
        self.ry0(qdev, wires=1)
        self.rz0(qdev, wires=3)
        self.crx0(qdev, wires=[0, 2])
        tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
        tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
        tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)


class HybridQuantumNAT(tq.QuantumModule):
    """Full quantum model that mirrors the classical HybridQuantumNAT.

    The model first encodes the input image into a 4‑qubit state using a
    Ry‑Rz‑xy encoder, applies the variational ``QLayer``, measures all qubits,
    and normalises the output with a 1‑D batch‑norm layer.
    """

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)

        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)

        self.q_layer(qdev)

        out = self.measure(qdev)
        return self.norm(out)


__all__ = ["HybridQuantumNAT"]
