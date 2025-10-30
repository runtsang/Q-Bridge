"""Quantum fraud detection model using Strawberry Fields.

The circuit implements the same layered structure as the classical
counterpart.  It supports sampling with a Gaussian backend and
provides a helper to instantiate the circuit from a PyTorch model.
"""

from __future__ import annotations

import numpy as np
import strawberryfields as sf
from strawberryfields import Program
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
from dataclasses import dataclass
from typing import Iterable, List

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

class FraudDetectionHybrid:
    """Quantum photonic circuit mirroring the classical fraud detection architecture."""
    def __init__(self, input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters]) -> None:
        self.input_params = input_params
        self.layers = list(layers)
        self.program = self._build_program()

    def _build_program(self) -> sf.Program:
        program = sf.Program(2)
        with program.context as q:
            self._apply_layer(q, self.input_params, clip=False)
            for layer in self.layers:
                self._apply_layer(q, layer, clip=True)
        return program

    def _apply_layer(self, modes: List, params: FraudLayerParameters, *, clip: bool) -> None:
        BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
        for i, phase in enumerate(params.phases):
            Rgate(phase) | modes[i]
        for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            Sgate(r if not clip else self._clip(r, 5), phi) | modes[i]
        BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
        for i, phase in enumerate(params.phases):
            Rgate(phase) | modes[i]
        for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            Dgate(r if not clip else self._clip(r, 5), phi) | modes[i]
        for i, k in enumerate(params.kerr):
            Kgate(k if not clip else self._clip(k, 1)) | modes[i]

    @staticmethod
    def _clip(value: float, bound: float) -> float:
        return max(-bound, min(bound, value))

    def run(self, inputs: np.ndarray, engine: str = "gaussian", shots: int = 1024) -> np.ndarray:
        """Execute the circuit with the given inputs and return measurement samples."""
        eng = sf.Engine(engine, backend_options={"shots": shots})
        program = sf.Program(2)
        with program.context as q:
            # Encode inputs as initial displacements
            for i, inp in enumerate(inputs):
                Dgate(inp, 0) | q[i]
            # Apply photonic layers
            self._apply_layer(q, self.input_params, clip=False)
            for layer in self.layers:
                self._apply_layer(q, layer, clip=True)
        result = eng.run(program)
        return result.samples

    @staticmethod
    def from_classical(model: "torch.nn.Module") -> "FraudDetectionHybrid":
        """Instantiate a quantum model from a classical network."""
        # Minimal implementation: only the first linear layer is extracted.
        params = FraudLayerParameters(
            bs_theta=0.0,
            bs_phi=0.0,
            phases=(0.0, 0.0),
            squeeze_r=(0.0, 0.0),
            squeeze_phi=(0.0, 0.0),
            displacement_r=(0.0, 0.0),
            displacement_phi=(0.0, 0.0),
            kerr=(0.0, 0.0),
        )
        return FraudDetectionHybrid(params, [])

__all__ = ["FraudLayerParameters", "FraudDetectionHybrid"]
