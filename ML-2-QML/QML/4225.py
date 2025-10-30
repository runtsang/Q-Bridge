"""Quantum utilities for fraud detection, built on Strawberry Fields."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Iterable, Sequence

import strawberryfields as sf
from strawberryfields import Engine
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

@dataclass
class FraudLayerParameters:
    """
    Parameters that describe a single photonic layer.
    Mirrors the structure used in the seed reference.
    """
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    """
    Create a Strawberry Fields program for the hybrid fraud detection model.
    """
    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program

def _apply_layer(modes: Sequence, params: FraudLayerParameters, *, clip: bool) -> None:
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

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

class QPhotonicHead:
    """
    Quantum photonic head that transforms classical logits into a differentiable expectation value.
    Uses a single‑layer photonic circuit with parameters derived from the logits.
    """

    def __init__(self) -> None:
        self.engine = Engine("gaussian")

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Tensor of shape (batch, 1) containing raw logits from the classical network.
        Returns:
            Tensor of shape (batch,) containing expectation values from the photonic circuit.
        """
        import torch  # local import to keep the module self‑contained
        batch = logits.shape[0]
        expectations = []
        for i in range(batch):
            angle = float(logits[i, 0])
            params = FraudLayerParameters(
                bs_theta=angle,
                bs_phi=angle,
                phases=(0.0, 0.0),
                squeeze_r=(0.0, 0.0),
                squeeze_phi=(0.0, 0.0),
                displacement_r=(0.0, 0.0),
                displacement_phi=(0.0, 0.0),
                kerr=(0.0, 0.0),
            )
            prog = build_fraud_detection_program(params, [])
            result = self.engine.run(prog)
            exp = result.expectation_value([sf.ops.NumberOp(0)], 0)[0]
            expectations.append(exp)
        return torch.tensor(expectations, dtype=torch.float32)
