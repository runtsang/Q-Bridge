from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, List

import numpy as np
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator


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


class FraudDetectionModel:
    """
    Quantum fraud‑detection model based on a layered photonic circuit.
    Each evaluation builds a new program with the supplied layer parameters
    and computes expectation values of the provided observables. Optional
    shot‑noise simulation is available.
    """

    def __init__(self, layers: Iterable[FraudLayerParameters]) -> None:
        self._layers = list(layers)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[FraudLayerParameters],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Return expectation values for each parameter set.

        Parameters
        ----------
        observables : iterable of qiskit BaseOperator
            Operators whose expectation values are computed on the circuit's statevector.
        parameter_sets : sequence of FraudLayerParameters
            The first element is treated as the input layer; the remaining
            elements correspond to the subsequent layers of the model.
        shots : int | None
            If provided, Gaussian noise with variance 1/shots is added to simulate
            finite‑shot sampling.
        seed : int | None
            Random seed for the noise generator.
        """
        rng = np.random.default_rng(seed) if shots is not None else None
        results: List[List[float]] = []

        for input_params in parameter_sets:
            prog = build_fraud_detection_program(input_params, self._layers)
            state = Statevector.from_instruction(prog)
            row = [state.expectation_value(obs) for obs in observables]
            if shots is not None:
                noise = rng.normal(0, 1 / np.sqrt(shots), size=len(row))
                row = (np.array(row) + noise).tolist()
            results.append(row)
        return results

    def sample(self, shots: int, seed: int | None = None) -> np.ndarray:
        """Generate raw samples from the circuit for the first layer."""
        rng = np.random.default_rng(seed)
        prog = build_fraud_detection_program(self._layers[0], self._layers[1:])
        state = Statevector.from_instruction(prog)
        probs = state.probs()
        return rng.choice(len(probs), size=shots, p=probs)


__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "FraudDetectionModel"]
