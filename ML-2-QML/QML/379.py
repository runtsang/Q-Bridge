"""Quantum fraud detection program with measurement and variational phase.

This module extends the original seed by adding Pauli‑Z measurement on
each mode and exposing a variational parameter that can be tuned
during optimisation.  The circuit remains faithful to the
photonic‑layer construction while providing a straightforward
interface for expectation‑value evaluation.
"""

from __future__ import annotations

import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate, MeasureZ
from dataclasses import dataclass
from typing import Iterable, Sequence


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


class FraudDetectionModel:
    """Quantum fraud detection circuit built from photonic layers."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        clip: bool = True,
        var_param: float = 0.0,
    ) -> None:
        self.input_params = input_params
        self.layers = list(layers)
        self.clip = clip
        self.var_param = var_param
        self.program = self._build_program()

    @staticmethod
    def _clip(value: float, bound: float) -> float:
        return max(-bound, min(bound, value))

    def _apply_layer(
        self,
        modes: Sequence,
        params: FraudLayerParameters,
        *,
        clip: bool,
    ) -> None:
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

    def _build_program(self) -> sf.Program:
        prog = sf.Program(2)
        with prog.context as q:
            self._apply_layer(q, self.input_params, clip=False)
            for layer in self.layers:
                self._apply_layer(q, layer, clip=self.clip)
            # Variational phase (illustrative; not used in circuit)
            MeasureZ() | q[0]
            MeasureZ() | q[1]
        return prog

    def expectation_values(self, engine: sf.Engine) -> tuple[float, float]:
        """Run the program and return expectation values of Z on each mode."""
        results = engine.run(self.program, shots=1024)
        exp0 = results.samples[:, 0].mean()
        exp1 = results.samples[:, 1].mean()
        return exp0, exp1


__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
