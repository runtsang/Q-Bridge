"""Quantum variant of the hybrid model using torchquantum."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from dataclasses import dataclass
from typing import Iterable, List, Optional

# ----------------------------------------------------------------------
# Fraud‑layer parameters (same dataclass as in the ML side)
# ----------------------------------------------------------------------
@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

# ----------------------------------------------------------------------
# Quantum module
# ----------------------------------------------------------------------
class QuantumNATHybrid(tq.QuantumModule):
    """Quantum encoder + variational layer that mimics the classical fraud‑detection stack."""
    class QLayer(tq.QuantumModule):
        def __init__(self, params: Optional[FraudLayerParameters] = None):
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
            self.params = params
            # Parameterised rotations
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            if self.params:
                # Map fraud‑parameters to rotation angles
                self.rx0(qdev, wires=0, params=self.params.bs_theta)
                self.ry0(qdev, wires=1, params=self.params.bs_phi)
                self.rz0(qdev, wires=3, params=self.params.displacement_r[0])
                self.crx0(qdev, wires=[0, 2], params=self.params.displacement_phi[0])
            else:
                self.rx0(qdev)
                self.ry0(qdev)
                self.rz0(qdev)
                self.crx0(qdev)
            # Fixed circuit primitives
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self, fraud_params: Optional[List[FraudLayerParameters]] = None):
        super().__init__()
        self.n_wires = 4
        # Encoder that embeds a 4‑dimensional feature vector
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        # Variational layer (one per fraud layer if provided)
        self.q_layers = nn.ModuleList(
            [self.QLayer(p) for p in fraud_params or [None]]
        )
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True
        )
        # Reduce the input image to a 4‑dim vector
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        # Apply each variational layer
        for layer in self.q_layers:
            layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

__all__ = ["QuantumNATHybrid", "FraudLayerParameters"]
