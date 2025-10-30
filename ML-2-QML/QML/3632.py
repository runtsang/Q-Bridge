from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
from strawberryfields import Sampler

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


def _apply_layer(modes: Sequence, params: FraudLayerParameters, *, clip: bool) -> None:
    """Insert a photonic layer into the program context."""
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


class FraudDetectionHybridQML:
    """Quantum‑classical hybrid that runs a photonic circuit and samples probabilities."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        backend: sf.Simulator = sf.Simulator("gaussian"),
    ) -> None:
        self.program = build_fraud_detection_program(input_params, layers)
        self.backend = backend
        self.sampler = Sampler(backend)

    def evaluate(self, num_shots: int = 1000) -> sf.Result:
        """Run the program and return sampled measurement outcomes."""
        return self.sampler.sample(self.program, shots=num_shots)

    def to_probabilities(self, shots: int = 1000) -> dict[tuple[int, int], float]:
        """Convert raw samples into a probability distribution over the two‑mode outcomes."""
        result = self.evaluate(num_shots=shots)
        probs = {}
        for outcome in result.samples:
            key = tuple(outcome)
            probs[key] = probs.get(key, 0) + 1
        total = sum(probs.values())
        for k in probs:
            probs[k] /= total
        return probs


__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "FraudDetectionHybridQML"]
