"""Quantum photonic fraud detection model with tunable measurement basis and variational post‑processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pennylane as qml
import torch
from torch import nn


@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer."""
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


def _apply_layer(
    modes: Sequence,
    params: FraudLayerParameters,
    *,
    clip: bool,
) -> None:
    """Apply a photonic layer to the given modes."""
    from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        Sgate(r if not clip else _clip(r, 5), phi) | modes[i]
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        Dgate(r if not clip else _clip(r, 5), phi) | modes[i]
    for i, k in enumerate(params.kerr):
        Kgate(k if not clip else _clip(k, 1)) | modes[i]


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> "sf.Program":
    """Create a Strawberry Fields program for the hybrid fraud detection model."""
    import strawberryfields as sf

    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program


class FraudDetection__gen343(nn.Module):
    """Quantum photonic fraud detection model with tunable measurement basis and variational post‑processing."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        measurement_basis: tuple[float, float] = (0.0, 0.0),
        postprocess_hidden_dim: int = 8,
    ) -> None:
        super().__init__()
        self.measurement_basis = measurement_basis
        # Build photonic program (kept for completeness)
        self.photonic_program = build_fraud_detection_program(input_params, layers)
        # Pennylane device
        self.dev = qml.device("default.qubit", wires=2)
        # Variational circuit
        self.qnode = qml.QNode(self._variational_circuit, self.dev)
        # Post‑processing network
        self.postprocess = nn.Sequential(
            nn.Linear(2, postprocess_hidden_dim),
            nn.ReLU(),
            nn.Linear(postprocess_hidden_dim, 1),
        )

    def _variational_circuit(self, x):
        """Variational circuit that maps 2‑dimensional input to expectation values."""
        # Input rotations
        qml.RY(x[0], wires=0)
        qml.RY(x[1], wires=1)
        # Entangling layer
        qml.CNOT(wires=[0, 1])
        # Tunable measurement basis
        qml.RZ(self.measurement_basis[0], wires=0)
        qml.RZ(self.measurement_basis[1], wires=1)
        # Measure expectation values of PauliZ
        return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: run variational circuit and post‑process."""
        inputs = x.detach().cpu().numpy()
        expectations = []
        for sample in inputs:
            exp = self.qnode(sample)
            expectations.append(exp)
        expectations = torch.tensor(expectations, dtype=torch.float32, device=x.device)
        preds = self.postprocess(expectations)
        return preds

    def set_measurement_basis(self, basis: tuple[float, float]) -> None:
        self.measurement_basis = basis


__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "FraudDetection__gen343",
]
