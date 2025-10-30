"""Quantum photonic fraud detection model with a variational layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate, Number
from strawberryfields import Engine


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
    """Clip a scalar to the interval [-bound, bound]."""
    return max(-bound, min(bound, value))


def _apply_layer(modes: Sequence, params: FraudLayerParameters, *, clip: bool) -> None:
    """Apply a photonic layer to the given modes."""
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


class FraudDetectionModel:
    """Quantum photonic fraud detection model with a tunable variational layer."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
    ) -> None:
        self.input_params = input_params
        self.layers = list(layers)
        self.var_params = np.zeros(3, dtype=float)  # [rot0, rot1, bs_theta]
        self.program = self._build_program()

    def _build_program(self) -> sf.Program:
        """Construct the Strawberry Fields program including a placeholder variational layer."""
        prog = sf.Program(2)
        with prog.context as q:
            _apply_layer(q, self.input_params, clip=False)
            for layer in self.layers:
                _apply_layer(q, layer, clip=True)
            # Variational layer: two tunable rotations followed by a beamsplitter
            Rgate(self.var_params[0]) | q[0]
            Rgate(self.var_params[1]) | q[1]
            BSgate(self.var_params[2], 0.0) | (q[0], q[1])
        return prog

    def set_variational_parameters(self, var_params: np.ndarray) -> None:
        """Set the tunable rotation angles and beamsplitter parameter."""
        if var_params.shape!= (3,):
            raise ValueError("Expected 3 variational parameters: two rotations and one beamsplitter angle.")
        self.var_params = var_params
        self.program = self._build_program()

    def expectation(self, engine: Engine, cutoff_dim: int = 5) -> float:
        """Run the program and return the photon‑number expectation in mode 0."""
        results = engine.run(self.program, cutoff_dim=cutoff_dim).state
        return results.expectation_value(Number(0))

    def sample(self, engine: Engine, shots: int = 1000, cutoff_dim: int = 5) -> np.ndarray:
        """Generate photon‑number samples from the program."""
        results = engine.run(self.program, shots=shots, cutoff_dim=cutoff_dim)
        return results.samples


__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
