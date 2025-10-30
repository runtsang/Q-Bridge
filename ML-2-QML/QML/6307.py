"""Quantum photonic fraud detection circuit with configurable backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate


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


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


class FraudDetectionEnhanced:
    """
    Build and evaluate a photonic fraud detection circuit.

    Parameters
    ----------
    input_params : FraudLayerParameters
        Parameters for the first layer (no clipping).
    layers : Iterable[FraudLayerParameters]
        Parameters for subsequent layers (clipped to a safe range).
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
    ):
        self.input_params = input_params
        self.layers = list(layers)

    def build_program(self) -> sf.Program:
        """Return a Strawberry Fields program representing the circuit."""
        prog = sf.Program(2)
        with prog.context as q:
            self._apply_layer(q, self.input_params, clip=False)
            for layer in self.layers:
                self._apply_layer(q, layer, clip=True)
        return prog

    def _apply_layer(
        self,
        modes: Sequence,
        params: FraudLayerParameters,
        *,
        clip: bool,
    ) -> None:
        """Apply a single photonic layer to the given modes."""
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

    def evaluate(
        self,
        backend: str | None = None,
        shots: int | None = None,
    ) -> dict:
        """
        Simulate the program on the specified backend.

        Parameters
        ----------
        backend : str, optional
            Name of the Strawberry Fields backend (e.g. 'gaussian',
            'tensor', 'gaussian_mixed').  If ``None`` the default
            Gaussian backend is used.
        shots : int, optional
            Number of measurement shots.  If ``None`` a state‑vector
            simulation is performed.
        Returns
        -------
        dict
            Dictionary containing the measurement statistics.
        """
        prog = self.build_program()
        if shots is None:
            eng = sf.Engine(backend=backend or "gaussian")
            results = eng.run(prog)
            return {"state": results.state}
        else:
            eng = sf.Engine(backend=backend or "gaussian")
            results = eng.run(prog, shots=shots)
            return {"shots": results.samples}

__all__ = ["FraudDetectionEnhanced", "FraudLayerParameters"]
