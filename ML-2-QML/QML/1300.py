"""Quantum fraud detection model using a rotation‑only variational ansatz."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import strawberryfields as sf
from strawberryfields import Program
from strawberryfields.ops import Rgate, Dgate, BSgate, Sgate, Kgate, Measure


@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]


class FraudDetectionHybrid:
    """Quantum fraud detection model that mirrors the classical architecture.
    Each layer is implemented as a small photonic circuit with a rotation‑only
    ansatz.  The parameters are shared with the classical side via the
    FraudLayerParameters dataclass."""
    def __init__(self,
                 input_params: FraudLayerParameters,
                 hidden_params: Iterable[FraudLayerParameters],
                 shots: int = 1024) -> None:
        self.program = self._build_program(input_params, hidden_params)
        self.shots = shots

    def _build_program(self,
                       input_params: FraudLayerParameters,
                       hidden_params: Iterable[FraudLayerParameters]) -> sf.Program:
        prog = sf.Program(2)
        with prog.context as q:
            self._apply_layer(q, input_params, clip=False)
            for params in hidden_params:
                self._apply_layer(q, params, clip=True)
            # Measure photon number in mode 0 as the output observable
            Measure | q[0]
        return prog

    def _apply_layer(self,
                     modes: Sequence,
                     params: FraudLayerParameters,
                     *,
                     clip: bool) -> None:
        # Simple rotation‑only ansatz: apply a single Rgate per mode
        for i, phase in enumerate(params.phases):
            Rgate(phase) | modes[i]
        # Add optional squeezing and displacement for richness
        for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            Sgate(r if not clip else self._clip(r, 5), phi) | modes[i]
        for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            Dgate(r if not clip else self._clip(r, 5), phi) | modes[i]
        # Kerr nonlinearity
        for i, k in enumerate(params.kerr):
            Kgate(k if not clip else self._clip(k, 1)) | modes[i]
        # Beam splitter to mix modes
        BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])

    @staticmethod
    def _clip(value: float, bound: float) -> float:
        return max(-bound, min(bound, value))

    def expectation(self, backend: str = "tf", device: str = "cpu") -> float:
        """Run the circuit on the chosen backend and return the mean photon number
        in mode 0.  The backend can be 'tf', 'fock', or 'gaussian'."""
        eng = sf.Engine(backend, backend_options={"device": device})
        results = eng.run(self.program, shots=self.shots)
        return results.meas[0].mean()

__all__ = ["FraudLayerParameters", "FraudDetectionHybrid"]
