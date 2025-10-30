"""Quantum version of FraudDetectionEnhanced with a parameter‑shared variational circuit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate


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
    return max(-bound, min(bound, value))


def _apply_layer(
    modes: Sequence,
    params: FraudLayerParameters,
    *,
    clip: bool,
) -> None:
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


def _apply_shared_variational_layer(
    modes: Sequence,
    shared_vars: dict,
    *,
    clip: bool,
) -> None:
    """Apply a variational layer that reuses the same set of parameters across all layers."""
    BSgate(shared_vars["theta"], shared_vars["phi"]) | (modes[0], modes[1])
    for i, phase in enumerate(shared_vars["phases"]):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(shared_vars["squeeze_r"], shared_vars["squeeze_phi"])):
        Sgate(r if not clip else _clip(r, 5), phi) | modes[i]
    BSgate(shared_vars["theta"], shared_vars["phi"]) | (modes[0], modes[1])
    for i, phase in enumerate(shared_vars["phases"]):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(shared_vars["displacement_r"], shared_vars["displacement_phi"])):
        Dgate(r if not clip else _clip(r, 5), phi) | modes[i]
    for i, k in enumerate(shared_vars["kerr"]):
        Kgate(k if not clip else _clip(k, 1)) | modes[i]


class FraudDetectionEnhanced:
    """Quantum fraud‑detection model that optionally uses a shared variational circuit."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        *,
        shared: bool = False,
    ) -> None:
        self.input_params = input_params
        self.layers = list(layers)
        self.shared = shared
        self.program = self._build_program()

    def _build_program(self) -> sf.Program:
        program = sf.Program(2)
        with program.context as q:
            _apply_layer(q, self.input_params, clip=False)
            if self.shared:
                shared_vars = {
                    "theta": sf.Variable("theta_shared"),
                    "phi": sf.Variable("phi_shared"),
                    "phases": (sf.Variable("phase0_shared"), sf.Variable("phase1_shared")),
                    "squeeze_r": (sf.Variable("squeeze_r0_shared"), sf.Variable("squeeze_r1_shared")),
                    "squeeze_phi": (sf.Variable("squeeze_phi0_shared"), sf.Variable("squeeze_phi1_shared")),
                    "displacement_r": (sf.Variable("disp_r0_shared"), sf.Variable("disp_r1_shared")),
                    "displacement_phi": (sf.Variable("disp_phi0_shared"), sf.Variable("disp_phi1_shared")),
                    "kerr": (sf.Variable("kerr0_shared"), sf.Variable("kerr1_shared")),
                }
                for _ in self.layers:
                    _apply_shared_variational_layer(q, shared_vars, clip=True)
            else:
                for layer in self.layers:
                    _apply_layer(q, layer, clip=True)
        return program

    def get_program(self) -> sf.Program:
        """Return the underlying Strawberry Fields program."""
        return self.program

    def regularization(self) -> float:
        """Return a simple L2 penalty over all shared parameters."""
        if not self.shared:
            return 0.0
        penalty = 0.0
        for var in self.program.variables.values():
            penalty += var.value().dot(var.value())
        return penalty


__all__ = ["FraudLayerParameters", "FraudDetectionEnhanced"]
