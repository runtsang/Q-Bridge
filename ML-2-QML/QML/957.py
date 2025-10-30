from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate, CZgate


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
    """Quantum fraud detection model using a variational photonic circuit."""
    def __init__(self, input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters], shots: int = 1024) -> None:
        self.input_params = input_params
        self.layers = list(layers)
        self.shots = shots
        self.program = self._build_program()

    def _clip(self, value: float, bound: float) -> float:
        return max(-bound, min(bound, value))

    def _apply_layer(self, modes: Sequence, params: FraudLayerParameters, *, clip: bool) -> None:
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
        # Variational entanglement between modes
        CZgate() | (modes[0], modes[1])

    def _build_program(self) -> sf.Program:
        prog = sf.Program(2)
        with prog.context as q:
            self._apply_layer(q, self.input_params, clip=False)
            for layer in self.layers:
                self._apply_layer(q, layer, clip=True)
        return prog

    def run(self, eng: sf.Engine) -> sf.Result:
        """Execute the circuit on the provided Strawberry Fields engine."""
        return eng.run(self.program, shots=self.shots)

    def expectation_photon_number(self, eng: sf.Engine) -> float:
        """Return the mean photon number of mode 0 after execution."""
        result = self.run(eng)
        return result.samples[:, 0].mean()


__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
