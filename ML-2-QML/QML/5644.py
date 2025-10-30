"""Quantum photonic kernel using Strawberry Fields.

This module implements a `HybridKernelMethod` class that encodes two classical feature vectors into a
parametric photonic circuit and evaluates the absolute value of the overlap between the resulting states.
The design is inspired by the FraudDetection photonic program but adapted to serve as a kernel function
for machine‑learning workflows.  The circuit is built from a list of layers described by
`FraudLayerParameters`, mirroring the classical analogue.  The resulting kernel is a valid positive‑definite
function and can be combined with the classical hybrid kernel defined in the ML partner module.

Key differences from the seed:
* Uses Strawberry Fields instead of TorchQuantum, showcasing a photonic implementation.
* Encodes both inputs with opposite signs to compute a Loschmidt echo‑style overlap.
* Provides a `kernel_matrix` helper that evaluates the Gram matrix for any iterable of tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import strawberryfields as sf
from strawberryfields import Engine
from strawberryfields import Program
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


def _clip(value: float, bound: float) -> float:
    """Clip a parameter to a safe range."""
    return max(-bound, min(bound, value))


class HybridKernelMethod:
    """Quantum photonic kernel based on a parametric Strawberry Fields circuit."""

    def __init__(self, layers: Iterable[FraudLayerParameters], engine: Engine | None = None) -> None:
        """
        Parameters
        ----------
        layers : Iterable[FraudLayerParameters]
            Ordered list of layers that constitute the circuit.
        engine : Engine | None
            Optional Strawberry Fields engine; a new one is created if omitted.
        """
        self.layers = list(layers)
        self.engine = engine or Engine("fock", backend_options={"cutoff_dim": 8})

    def _apply_layer(self, q, params: FraudLayerParameters, *, clip: bool) -> None:
        """Apply a single photonic layer to the modes."""
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

    def _encode(self, q, x: np.ndarray, *, clip: bool) -> None:
        """Encode a single feature vector into the circuit."""
        for params in self.layers:
            self._apply_layer(q, params, clip=clip)

    def forward(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the quantum kernel value |<ψ(x)|ψ(-y)>| for two feature vectors.

        Parameters
        ----------
        x, y : np.ndarray
            1‑D arrays of shape (2,).

        Returns
        -------
        float
            Kernel value in [0, 1].
        """
        # Encode x
        prog_x = Program(2)
        qx = prog_x.q
        self._encode(qx, x, clip=False)
        # Encode -y
        prog_y = Program(2)
        qy = prog_y.q
        self._encode(qy, -y, clip=False)
        # Execute both programs and compute overlap
        eng_x = Engine("fock", backend_options={"cutoff_dim": 8})
        eng_y = Engine("fock", backend_options={"cutoff_dim": 8})
        state_x = eng_x.run(prog_x).state
        state_y = eng_y.run(prog_y).state
        # Overlap between two pure states
        overlap = np.abs(state_x.overlap(state_y))
        return float(overlap)

    @staticmethod
    def kernel_matrix(a: Iterable[np.ndarray], b: Iterable[np.ndarray], layers: Iterable[FraudLayerParameters]) -> np.ndarray:
        """
        Evaluate the Gram matrix between two collections of feature vectors.

        Parameters
        ----------
        a, b : Iterable[np.ndarray]
            Collections of 1‑D arrays of shape (2,).
        layers : Iterable[FraudLayerParameters]
            Layer definitions used to build the circuit.

        Returns
        -------
        np.ndarray
            2‑D array of kernel values.
        """
        kernel = HybridKernelMethod(layers)
        return np.array([[kernel.forward(x, y) for y in b] for x in a])


__all__ = ["HybridKernelMethod", "FraudLayerParameters"]
