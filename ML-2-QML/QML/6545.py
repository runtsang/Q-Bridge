"""
Quantum kernel implemented with a photonic variational circuit inspired
by the fraud‑detection photonic layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, List

import torch
import numpy as np
import strawberryfields as sf
from strawberryfields.ops import BSgate, Rgate, Sgate, Dgate, Kgate


@dataclass
class FraudLayerParameters:
    """Parameters that describe a single photonic layer."""
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
    q: Sequence[object],
    params: FraudLayerParameters,
    *,
    clip: bool,
) -> None:
    """Apply a single photonic layer to the given program."""
    BSgate(params.bs_theta, params.bs_phi) | (q[0], q[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | q[i]
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        Sgate(r if not clip else _clip(r, 5), phi) | q[i]
    BSgate(params.bs_theta, params.bs_phi) | (q[0], q[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | q[i]
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        Dgate(r if not clip else _clip(r, 5), phi) | q[i]
    for i, k in enumerate(params.kerr):
        Kgate(k if not clip else _clip(k, 1)) | q[i]


def _build_base_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    """Create a program that contains all photonic layers but no data encoding."""
    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program


class FraudQuantumKernel:
    """
    Quantum kernel that evaluates the overlap of photonic states generated
    by a fraud‑detection inspired circuit.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters] | None = None,
        backend_name: str = "gaussian_state",
    ) -> None:
        self.input_params = input_params
        self.layers = list(layers or [])
        self.backend_name = backend_name
        self.base_program = _build_base_program(self.input_params, self.layers)

    def _simulate(self, program: sf.Program) -> np.ndarray:
        eng = sf.Engine(self.backend_name)
        result = eng.run(program)
        return result.state.get_state_vector()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Compute the quantum kernel value between two feature vectors.
        Each vector is encoded via displacement gates before the photonic layers.
        """
        # Create two independent copies of the base program
        prog_x = self.base_program.copy()
        prog_y = self.base_program.copy()

        # Encode data into displacement gates
        with prog_x.context as q:
            for i, xi in enumerate(x.numpy()):
                Dgate(xi, 0.0) | q[i]
        with prog_y.context as q:
            for i, yi in enumerate(y.numpy()):
                Dgate(yi, 0.0) | q[i]

        # Run simulation and obtain state vectors
        state_x = self._simulate(prog_x)
        state_y = self._simulate(prog_y)

        # Overlap magnitude squared
        overlap = abs(np.vdot(state_x, state_y)) ** 2
        return torch.tensor(overlap, dtype=torch.float32)

    def kernel_matrix(
        self,
        a: Sequence[torch.Tensor],
        b: Sequence[torch.Tensor],
    ) -> np.ndarray:
        """Return the Gram matrix for two collections of vectors."""
        return np.array(
            [[self.forward(x, y).item() for y in b] for x in a]
        )


__all__ = ["FraudLayerParameters", "FraudQuantumKernel"]
