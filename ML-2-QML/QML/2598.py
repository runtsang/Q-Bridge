"""Quantum‑augmented fraud detection model using a photonic‑inspired variational circuit.

The circuit is built by translating the classical photonic parameters into
parameterised gates in TorchQuantum.  A random layer injects additional
expressivity, and the final measurement is linearly mapped to a scalar
prediction.  The model can be trained end‑to‑end using PyTorch loss
functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torchquantum as tq


@dataclass
class FraudLayerParameters:
    """Parameters that control a single photonic‑style block in the quantum circuit."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


class PhotonicQuantumLayer(tq.QuantumModule):
    """A single block that emulates a photonic layer using parameterised gates."""
    def __init__(self, params: FraudLayerParameters):
        super().__init__()
        self.params = params
        # Parameterised single‑qubit rotations that mimic beam‑splitter and phase
        # shifts.  The angles are fixed by the supplied parameters but the gates
        # themselves remain trainable to allow fine‑tuning during optimisation.
        self.rgate0 = tq.RY(has_params=True, trainable=True)
        self.rgate1 = tq.RY(has_params=True, trainable=True)
        self.rz0 = tq.RZ(has_params=True, trainable=True)
        self.rz1 = tq.RZ(has_params=True, trainable=True)
        # Random layer to increase expressivity
        self.random = tq.RandomLayer(n_ops=10, wires=list(range(2)))

    def forward(self, qdev: tq.QuantumDevice) -> None:
        # Beam‑splitter analogue via two RY rotations
        self.rgate0(qdev, wires=0, params=self.params.bs_theta)
        self.rgate1(qdev, wires=1, params=self.params.bs_phi)
        # Phase shifts
        self.rz0(qdev, wires=0, params=self.params.phases[0])
        self.rz1(qdev, wires=1, params=self.params.phases[1])
        # Squeezing and displacement are emulated by additional rotations
        self.rgate0(qdev, wires=0, params=self.params.squeeze_r[0])
        self.rgate1(qdev, wires=1, params=self.params.squeeze_r[1])
        # Random layer
        self.random(qdev)


class FraudQuantumDetector(tq.QuantumModule):
    """Full quantum fraud detection circuit."""
    def __init__(self, input_params: FraudLayerParameters, layers_params: Sequence[FraudLayerParameters]):
        super().__init__()
        self.n_wires = 2
        # Encoder that maps classical data into the computational basis
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{self.n_wires}xRy"])
        # Photonic layers
        self.layers = nn.ModuleList([PhotonicQuantumLayer(p) for p in layers_params])
        # Measurement
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Classical head
        self.head = nn.Linear(self.n_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        # Encode classical features
        self.encoder(qdev, state_batch)
        # Apply photonic layers
        for layer in self.layers:
            layer(qdev)
        # Measure
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)


__all__ = ["FraudLayerParameters", "PhotonicQuantumLayer", "FraudQuantumDetector"]
