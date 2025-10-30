import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
from dataclasses import dataclass
from typing import Iterable, List

@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

class FraudDetectionAdvanced:
    """
    Quantum fraud detection model built with a depth‑controlled ladder of
    photonic operations. Parameters are shared across layers, enabling
    systematic scaling of circuit depth.
    """
    def __init__(self,
                 input_params: FraudLayerParameters,
                 depth: int = 3,
                 clip_bounds: float = 5.0):
        self.input_params = input_params
        self.depth = depth
        self.clip_bounds = clip_bounds
        self.program = self._build_program()

    def _clip(self, value: float, bound: float) -> float:
        return max(-bound, min(bound, value))

    def _apply_layer(self, q: sf.Program.Context, params: FraudLayerParameters, clip: bool):
        # Beam splitter
        BSgate(params.bs_theta, params.bs_phi) | (q[0], q[1])
        # Phase shifts
        for i, phase in enumerate(params.phases):
            Rgate(phase) | q[i]
        # Squeezing
        for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            Sgate(r if not clip else self._clip(r, self.clip_bounds), phi) | q[i]
        # Beam splitter again
        BSgate(params.bs_theta, params.bs_phi) | (q[0], q[1])
        # Phase shifts again
        for i, phase in enumerate(params.phases):
            Rgate(phase) | q[i]
        # Displacement
        for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            Dgate(r if not clip else self._clip(r, self.clip_bounds), phi) | q[i]
        # Kerr
        for i, k in enumerate(params.kerr):
            Kgate(k if not clip else self._clip(k, 1.0)) | q[i]

    def _build_program(self) -> sf.Program:
        prog = sf.Program(2)
        with prog.context as q:
            # first layer without clipping
            self._apply_layer(q, self.input_params, clip=False)
            # repeated layers with clipping
            for _ in range(self.depth):
                self._apply_layer(q, self.input_params, clip=True)
        return prog

    def get_program(self) -> sf.Program:
        return self.program

    def simulate(self, shots: int = 1024) -> sf.Result:
        eng = sf.Engine("gaussian")
        return eng.run(self.program, shots=shots)

__all__ = ["FraudLayerParameters", "FraudDetectionAdvanced"]
