"""Hybrid fraud detection model – quantum photonic implementation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate


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


class FraudDetectionModel:
    """Quantum photonic fraud‑detection model with sampling support."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        *,
        clip: bool = True,
        backend: str = "gaussian",
    ) -> None:
        self.input_params = input_params
        self.layers = list(layers)
        self.clip = clip
        self.backend = backend

    def _clip(self, value: float, bound: float) -> float:
        return max(-bound, min(bound, value))

    def _apply_layer(self, modes: Sequence, params: FraudLayerParameters, clip: bool) -> None:
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

    def build_quantum(self) -> sf.Program:
        """Create the Strawberry Fields program for the hybrid model."""
        program = sf.Program(2)
        with program.context as q:
            self._apply_layer(q, self.input_params, clip=False)
            for layer in self.layers:
                self._apply_layer(q, layer, clip=self.clip)
        return program

    def sample(self, shots: int = 1000, seed: int = 42):
        """Run the program and return sampled measurement outcomes."""
        prog = self.build_quantum()
        eng = sf.Engine(self.backend, backend_options={"seed": seed})
        result = eng.run(prog, shots=shots)
        return result.samples

    def expectation(
        self,
        observable: sf.ops.Observable,
        shots: int = 1000,
        seed: int = 42,
    ):
        """Compute the expectation value of a given observable."""
        prog = self.build_quantum()
        eng = sf.Engine(self.backend, backend_options={"seed": seed})
        result = eng.run(prog, shots=shots)
        return result.expectation(observable)

__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
