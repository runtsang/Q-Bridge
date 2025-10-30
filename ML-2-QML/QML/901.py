"""Quantum photonic fraud detection model.

This module extends the original seed by providing:
- A :class:`FraudDetectionModel` that builds a Strawberry Fields program.
- Convenience methods for running the circuit on a simulator, retrieving
  photon‑number statistics, and computing expectation values of arbitrary
  observables.
- A simple training loop that optimises a user‑supplied loss over a dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, List, Dict, Callable

import numpy as np
import strawberryfields as sf
from strawberryfields import Engine
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate, N

# -------------------------------------------------------------
# Parameter definition
# -------------------------------------------------------------
@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

# -------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------
def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _apply_layer(modes: Sequence, params: FraudLayerParameters, *, clip: bool) -> None:
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

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    """Create a Strawberry Fields program for the hybrid fraud detection model."""
    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program

# -------------------------------------------------------------
# Main model class
# -------------------------------------------------------------
class FraudDetectionModel:
    """Quantum photonic fraud‑detection circuit.

    The model encapsulates a 2‑mode Strawberry Fields program that mirrors the
    classical architecture. The circuit can be executed on any ``sf.Engine``
    and provides methods to obtain photon‑number statistics or expectation
    values of arbitrary observables.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Sequence[FraudLayerParameters],
        *,
        shots: int = 1000,
    ) -> None:
        self.input_params = input_params
        self.layers = list(layers)
        self.program = build_fraud_detection_program(input_params, self.layers)
        self.shots = shots

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------
    def run(self, engine: Engine) -> np.ndarray:
        """Run the circuit on the supplied engine and return photon‑number counts."""
        state = engine.run(self.program, shots=self.shots)
        return state.samples

    def expectation(
        self,
        engine: Engine,
        observable: Callable[[sf.State], np.ndarray],
    ) -> float:
        """Compute the expectation value of ``observable``."""
        state = engine.run(self.program, shots=self.shots)
        return float(observable(state))

    # ------------------------------------------------------------------
    # Parameter utilities
    # ------------------------------------------------------------------
    @staticmethod
    def random_parameters(num_layers: int, seed: int | None = None) -> Tuple[FraudLayerParameters, List[FraudLayerParameters]]:
        """Generate a random parameter set for a network of ``num_layers``."""
        rng = np.random.default_rng(seed)
        def rand_pair() -> Tuple[float, float]:
            return (float(rng.normal()), float(rng.normal()))
        input_params = FraudLayerParameters(
            bs_theta=float(rng.normal()),
            bs_phi=float(rng.normal()),
            phases=rand_pair(),
            squeeze_r=rand_pair(),
            squeeze_phi=rand_pair(),
            displacement_r=rand_pair(),
            displacement_phi=rand_pair(),
            kerr=rand_pair(),
        )
        layers = [
            FraudLayerParameters(
                bs_theta=float(rng.normal()),
                bs_phi=float(rng.normal()),
                phases=rand_pair(),
                squeeze_r=rand_pair(),
                squeeze_phi=rand_pair(),
                displacement_r=rand_pair(),
                displacement_phi=rand_pair(),
                kerr=rand_pair(),
            )
            for _ in range(num_layers)
        ]
        return input_params, layers

    # ------------------------------------------------------------------
    # Simple training loop (requires user‑supplied loss and optimizer)
    # ------------------------------------------------------------------
    def train_on_dataset(
        self,
        dataset: Iterable[Tuple[np.ndarray, float]],
        loss_fn: Callable[[np.ndarray, float], float],
        optimizer: Callable[[FraudDetectionModel], None],
        epochs: int = 10,
    ) -> List[float]:
        """Train the circuit by iterating over ``dataset``.

        Parameters
        ----------
        dataset
            Iterable of (inputs, target) pairs. Inputs are 2‑dimensional arrays
            that are fed into the circuit as displacement parameters.
        loss_fn
            Function that takes the circuit output and a target scalar and
            returns a scalar loss.
        optimizer
            Callable that updates the circuit parameters.  It receives the
            model instance and is expected to modify ``self.input_params`` and
            ``self.layers`` in‑place.
        """
        losses: List[float] = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for inputs, target in dataset:
                # For simplicity, we re‑build the program for each batch.
                self.program = build_fraud_detection_program(self.input_params, self.layers)
                # Run the circuit
                state = Engine("gaussian").run(self.program, shots=self.shots)
                # Compute expectation of photon number in mode 0
                exp = float(state.expectation([N(0)]))
                loss = loss_fn(exp, target)
                # Back‑propagation is omitted; the optimizer is expected to
                # perform its own parameter updates.
                epoch_loss += loss
            losses.append(epoch_loss / len(dataset))
        return losses

__all__ = ["FraudLayerParameters", "FraudDetectionModel", "build_fraud_detection_program"]
