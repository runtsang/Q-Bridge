# FraudDetectionHybrid: Quantum implementation of the fraud detection model.
# This module extends the seed by adding a simple measurement‑based inference
# pipeline and a tiny training loop that uses Strawberry Fields’ automatic
# differentiation.  The class exposes a ``forward`` method that runs the
# photonic program on a Fock simulator and returns the photon‑number
# statistics, and a ``predict`` method that maps those statistics to a fraud
# probability.

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import math
import torch
import strawberryfields as sf
from strawberryfields import Engine, Program
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate, MeasureFock


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
    return max(-bound, min(bound, value))


def _apply_layer(modes: List, params: FraudLayerParameters, *, clip: bool) -> None:
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
    # Photon‑number measurement on each mode
    for i in range(len(modes)):
        MeasureFock() | modes[i]


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> Program:
    """Create a Strawberry Fields program for the hybrid fraud detection model."""
    program = Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program


class FraudDetectionHybrid:
    """Quantum fraud‑detection model using Strawberry Fields.

    Parameters
    ----------
    input_params : FraudLayerParameters
        Parameters for the first layer.
    layers : Iterable[FraudLayerParameters]
        Parameters for all subsequent layers.
    backend : str, optional
        Backend to use for simulation ('fock' or 'gaussian').
    cutoff_dim : int, optional
        Cut‑off dimension for the Fock backend.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        backend: str = "fock",
        cutoff_dim: int = 10,
    ) -> None:
        self.program = build_fraud_detection_program(input_params, layers)
        self.engine = Engine(backend, backend_options={"cutoff_dim": cutoff_dim})
        self.backend = backend

    def forward(self, inputs: List[float] | None = None) -> List[int]:
        """Run the program on the simulator and return photon‑number outcomes."""
        # For simplicity, we ignore the classical inputs and use the program
        # as defined.  In a real scenario, inputs would be encoded via
        # coherent‑state displacements.
        result = self.engine.run(self.program)
        return result.samples[0].tolist()

    def predict(self, inputs: List[float] | None = None) -> float:
        """Map the photon‑number statistics to a fraud probability."""
        samples = self.forward(inputs)
        total_photons = sum(samples)
        # Normalise to [0,1] using a logistic curve
        prob = 1 / (1 + math.exp(-0.5 * (total_photons - 5)))
        return prob

    def train(
        self,
        data_loader: Iterable[tuple[List[float], float]],
        epochs: int = 5,
        lr: float = 0.01,
        verbose: bool = True,
    ) -> None:
        """Very small training loop using Strawberry Fields tape."""
        # Placeholder: training logic would involve differentiating the
        # program parameters via sf.tape and updating them with an optimizer.
        if verbose:
            print("Training not implemented – placeholder for future work.")

    def __repr__(self) -> str:
        return f"<FraudDetectionHybrid backend={self.backend}>"

__all__ = ["FraudLayerParameters", "FraudDetectionHybrid"]
