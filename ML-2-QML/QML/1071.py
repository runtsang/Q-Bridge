"""Variational photonic circuit for fraud detection with measurement support."""
from __future__ import annotations

import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

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

class FraudDetectionHybrid:
    """Variational photonic circuit for fraud detection with optional measurement."""
    def __init__(self,
                 input_params: FraudLayerParameters,
                 layers: Iterable[FraudLayerParameters],
                 measurement: bool = True,
                 clip: bool = True) -> None:
        self.input_params = input_params
        self.layers = list(layers)
        self.measurement = measurement
        self.clip = clip
        self.program = self._build_program()

    def _apply_layer(self,
                     q: Sequence,
                     params: FraudLayerParameters,
                     *,
                     clip: bool) -> None:
        BSgate(params.bs_theta, params.bs_phi) | (q[0], q[1])
        for i, phase in enumerate(params.phases):
            Rgate(phase) | q[i]
        for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            Sgate(r if not clip else self._clip(r, 5), phi) | q[i]
        BSgate(params.bs_theta, params.bs_phi) | (q[0], q[1])
        for i, phase in enumerate(params.phases):
            Rgate(phase) | q[i]
        for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            Dgate(r if not clip else self._clip(r, 5), phi) | q[i]
        for i, k in enumerate(params.kerr):
            Kgate(k if not clip else self._clip(k, 1)) | q[i]

    def _clip(self, value: float, bound: float) -> float:
        return max(-bound, min(bound, value))

    def _build_program(self) -> sf.Program:
        prog = sf.Program(2)
        with prog.context as q:
            self._apply_layer(q, self.input_params, clip=False)
            for layer in self.layers:
                self._apply_layer(q, layer, clip=self.clip)
            if self.measurement:
                # Measure photon number on both modes
                sf.ops.MeasureFock() | q[0]
                sf.ops.MeasureFock() | q[1]
        return prog

    def run(self, backend: sf.backends.Backend, shots: int = 1024) -> sf.Result:
        return backend.run(self.program, shots=shots)
