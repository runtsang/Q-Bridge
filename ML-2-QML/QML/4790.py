"""Hybrid quantum model that mirrors the classical pipeline with QML primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, List

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf


# ----------------------------------------------------------------------
# Fraud‑Detection style parameters (re‑used in quantum layer)
# ----------------------------------------------------------------------
@dataclass
class FraudLayerParameters:
    """Parameters describing a photonic layer, reused for the quantum analogue."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


# ----------------------------------------------------------------------
# Custom quantum layers
# ----------------------------------------------------------------------
class ConvLayer(tq.QuantumModule):
    """Parameterized 2‑qubit convolution block inspired by the QCNN conv_circuit."""

    def __init__(self, params: Sequence[float]):
        super().__init__()
        assert len(params) == 3, "ConvLayer expects 3 parameters."
        self.params = torch.tensor(params, dtype=torch.float32)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice):
        p = self.params
        tq.RZ(p[0], wires=0, static=self.static_mode, parent_graph=self.graph)
        tq.CX(wires=[0, 1], static=self.static_mode, parent_graph=self.graph)
        tq.RZ(p[1], wires=0, static=self.static_mode, parent_graph=self.graph)
        tq.RY(p[2], wires=1, static=self.static_mode, parent_graph=self.graph)
        tq.CX(wires=[0, 1], static=self.static_mode, parent_graph=self.graph)


class PoolLayer(tq.QuantumModule):
    """Parameterized 2‑qubit pooling block."""

    def __init__(self, params: Sequence[float]):
        super().__init__()
        assert len(params) == 3, "PoolLayer expects 3 parameters."
        self.params = torch.tensor(params, dtype=torch.float32)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice):
        p = self.params
        tq.RZ(p[0], wires=0, static=self.static_mode, parent_graph=self.graph)
        tq.CX(wires=[0, 1], static=self.static_mode, parent_graph=self.graph)
        tq.RZ(p[1], wires=0, static=self.static_mode, parent_graph=self.graph)
        tq.RY(p[2], wires=1, static=self.static_mode, parent_graph=self.graph)


class FraudQuantumLayer(tq.QuantumModule):
    """Quantum analogue of the photonic fraud layer using standard gates."""

    def __init__(self, params: FraudLayerParameters):
        super().__init__()
        self.params = params
        self.n_wires = 2  # operate on 2 qubits for simplicity

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice):
        # Randomness layer
        tq.RandomLayer(n_ops=10, wires=list(range(self.n_wires)))(qdev)
        # Parameterized rotations
        tq.RX(self.params.bs_theta, wires=0, static=self.static_mode, parent_graph=self.graph)
        tq.RY(self.params.bs_phi, wires=1, static=self.static_mode, parent_graph=self.graph)
        # Two‑qubit entanglement
        tq.CRX(self.params.bs_theta, wires=[0, 1], static=self.static_mode, parent_graph=self.graph)
        # Additional single‑qubit rotations from phases
        tq.RZ(self.params.phases[0], wires=0, static=self.static_mode, parent_graph=self.graph)
        tq.RZ(self.params.phases[1], wires=1, static=self.static_mode, parent_graph=self.graph)
        # Optional squeezing‑like operations via RX/RZ
        tq.RX(self.params.squeeze_r[0], wires=0, static=self.static_mode, parent_graph=self.graph)
        tq.RZ(self.params.squeeze_phi[0], wires=0, static=self.static_mode, parent_graph=self.graph)
        tq.RX(self.params.squeeze_r[1], wires=1, static=self.static_mode, parent_graph=self.graph)
        tq.RZ(self.params.squeeze_phi[1], wires=1, static=self.static_mode, parent_graph=self.graph)
        # Displacement‑like rotations
        tq.RY(self.params.displacement_r[0], wires=0, static=self.static_mode, parent_graph=self.graph)
        tq.RZ(self.params.displacement_phi[0], wires=0, static=self.static_mode, parent_graph=self.graph)
        tq.RY(self.params.displacement_r[1], wires=1, static=self.static_mode, parent_graph=self.graph)
        tq.RZ(self.params.displacement_phi[1], wires=1, static=self.static_mode, parent_graph=self.graph)
        # Kerr‑like phase‑shift
        tq.RZ(self.params.kerr[0], wires=0, static=self.static_mode, parent_graph=self.graph)
        tq.RZ(self.params.kerr[1], wires=1, static=self.static_mode, parent_graph=self.graph)


# ----------------------------------------------------------------------
# Main hybrid quantum model
# ----------------------------------------------------------------------
class QuantumHybridModel(tq.QuantumModule):
    """Hybrid quantum architecture featuring encoder, conv/pool layers, and a fraud layer."""

    def __init__(
        self,
        fraud_params: FraudLayerParameters,
        conv_params: Sequence[float],
        pool_params: Sequence[float],
    ) -> None:
        super().__init__()
        self.n_wires = 4
        # Feature‑map encoder (4‑qubit 4x4_ryzxy)
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        # Custom convolution, pooling, and fraud modules
        self.conv_layer = ConvLayer(conv_params)
        self.pool_layer = PoolLayer(pool_params)
        self.fraud_layer = FraudQuantumLayer(fraud_params)
        # Measurement and normalization
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Batch of images of shape (B, 1, 28, 28) on the same device as the qubits.
        Returns:
            Normalized expectation values per batch item.
        """
        bsz = x.shape[0]
        # Simple spatial pooling to 16‑dim feature vector
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        # Quantum device
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True
        )
        # Encode data
        self.encoder(qdev, pooled)
        # Apply layers
        self.conv_layer(qdev)
        self.pool_layer(qdev)
        self.fraud_layer(qdev)
        # Measurement
        out = self.measure(qdev)
        return self.norm(out)


__all__ = [
    "FraudLayerParameters",
    "ConvLayer",
    "PoolLayer",
    "FraudQuantumLayer",
    "QuantumHybridModel",
]
