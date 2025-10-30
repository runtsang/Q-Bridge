"""Quantum implementation of the fraud detection circuit using Strawberry Fields.

The circuit mirrors the classical architecture, applying a sequence of
beam‑splitter, squeezing, rotation, displacement and Kerr gates.
After the circuit the photon‑number operator on mode 0 is measured and
returned as a fraud score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import strawberryfields as sf
from strawberryfields import Engine
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate, MeasureFock


@dataclass
class FraudLayerParameters:
    """Parameters for one photonic layer, matching the classical counterpart."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


class FraudDetectionModel:
    """Hybrid photonic circuit for fraud detection.

    The circuit is constructed from an input layer followed by a stack
    of configurable layers.  Each layer applies a sequence of beam‑splitter,
    squeezing, rotation, displacement and Kerr gates, mimicking the
    structure of the classical model.  After the circuit the photon‑number
    operator is measured on mode 0 to generate a fraud probability.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layer_params: Iterable[FraudLayerParameters],
        *,
        clip: bool = True,
    ) -> None:
        self.program = sf.Program(2)
        self.clip = clip
        with self.program.context as q:
            self._apply_layer(q, input_params, clip=False)
            for p in layer_params:
                self._apply_layer(q, p, clip=clip)
            # measurement
            MeasureFock() | q[0]
            MeasureFock() | q[1]

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
            r_adj = r if not clip else _clip(r, 5)
            Sgate(r_adj, phi) | modes[i]
        BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
        for i, phase in enumerate(params.phases):
            Rgate(phase) | modes[i]
        for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            r_adj = r if not clip else _clip(r, 5)
            Dgate(r_adj, phi) | modes[i]
        for i, k in enumerate(params.kerr):
            k_adj = k if not clip else _clip(k, 1)
            Kgate(k_adj) | modes[i]

    def evaluate(self, dev: Engine | None = None) -> float:
        """Run the circuit and return the expectation value of n on mode 0."""
        if dev is None:
            dev = Engine("gaussian", backend="fock", cutoff_dim=10)
        state = dev.run(self.program)
        # photon number expectation for mode 0
        exp_n = state.expectation_value("n", 0)
        return float(exp_n)


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


__all__ = ["FraudDetectionModel"]
