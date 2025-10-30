"""Variational photonic fraud detection model using Strawberry Fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import strawberryfields as sf
from strawberryfields.ops import (
    BSgate,
    Dgate,
    Kgate,
    Rgate,
    Sgate,
    MeasureZ,
)

@dataclass
class FraudLayerParameters:
    """Parameters for a single photonic layer."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    """Clip a scalar to the range [-bound, bound]."""
    return max(-bound, min(bound, value))


class FraudDetectionEnhanced:
    """
    Variational Strawberry Fields circuit that mirrors the classical architecture.

    Parameters
    ----------
    input_params : FraudLayerParameters
        Parameters for the initial (unclipped) layer.
    layers : Iterable[FraudLayerParameters]
        Parameters for the subsequent layers; each will have its parameters clipped.
    shots : int, optional
        Number of measurement shots for expectation estimation.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        *,
        shots: int = 1024,
    ) -> None:
        self.input_params = input_params
        self.layers = list(layers)
        self.shots = shots
        self.program = self._build_program()

    def _build_program(self) -> sf.Program:
        prog = sf.Program(2)
        with prog.context as q:
            # Encode the input features into the first mode
            _apply_layer(q, self.input_params, clip=False)
            for layer in self.layers:
                _apply_layer(q, layer, clip=True)
            # Measure both modes
            MeasureZ | q[0]
            MeasureZ | q[1]
        return prog

    def _apply_layer(self, modes: Sequence, params: FraudLayerParameters, *, clip: bool) -> None:
        # Beam splitter
        BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
        # Phase rotations
        for i, phase in enumerate(params.phases):
            Rgate(phase) | modes[i]
        # Squeezing (clipped if required)
        for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            Sgate(_clip(r, 5.0) if clip else r, phi) | modes[i]
        # Second beam splitter
        BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
        # Phase rotations again
        for i, phase in enumerate(params.phases):
            Rgate(phase) | modes[i]
        # Displacement (clipped if required)
        for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            Dgate(_clip(r, 5.0) if clip else r, phi) | modes[i]
        # Kerr nonlinearity (clipped if required)
        for i, k in enumerate(params.kerr):
            Kgate(_clip(k, 1.0) if clip else k) | modes[i]

    def sample(self, engine: sf.Engine) -> tuple[float, float]:
        """Run the circuit on the given engine and return expectation values."""
        results = engine.run(self.program, shots=self.shots).samples
        # Expectation value of Z for each mode
        exp_z = results.mean(axis=0)
        return tuple(exp_z)

    def get_shots(self) -> int:
        """Return the number of shots used for measurement."""
        return self.shots


__all__ = ["FraudLayerParameters", "FraudDetectionEnhanced"]
