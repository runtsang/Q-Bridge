"""Quantum photonic fraud detection model with amplitude encoding and measurement readout."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate, MeasureZ


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


class FraudDetectionModel:
    """Quantum photonic fraud detection model with amplitude encoding and measurement readout.

    The circuit mirrors the classical layers but adds additional entanglement and
    a measurement operator to produce a fraud probability.
    """

    def __init__(self, params_list: List[FraudLayerParameters], shots: int = 1000) -> None:
        self.params_list = params_list
        self.shots = shots

    def _apply_layer(self, modes: List, params: FraudLayerParameters, *, clip: bool) -> None:
        BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
        for i, phase in enumerate(params.phases):
            Rgate(phase) | modes[i]
        for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            Sgate(r if not clip else self._clip(r, 5), phi) | modes[i]
        BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
        for i, phase in enumerate(params.phases):
            Rgate(phase) | modes[i]
        for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            Dgate(r if not clip else self._clip(r, 5), phi) | modes[i]
        for i, k in enumerate(params.kerr):
            Kgate(k if not clip else self._clip(k, 1)) | modes[i]

    @staticmethod
    def _clip(value: float, bound: float) -> float:
        return max(-bound, min(bound, value))

    def evaluate(self, input_vector: List[float]) -> float:
        """Run the circuit with the given input vector and return
        the probability of fraud (photon detection in mode 0)."""
        prog = sf.Program(2)
        with prog.context as q:
            # Encode inputs as displacements before the first layer
            for i, val in enumerate(input_vector):
                Dgate(val, 0) | q[i]
            for i, params in enumerate(self.params_list):
                self._apply_layer(q, params, clip=i!= 0)
            MeasureZ | q[0]
        eng = sf.Engine("fock", backend_options={"cutoff_dim": 10})
        result = eng.run(prog, shots=self.shots)
        counts = result.counts
        fraud_counts = sum(
            count for state, count in counts.items() if state[0] > 0
        )
        return fraud_counts / self.shots
