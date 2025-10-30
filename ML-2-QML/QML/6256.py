import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
from dataclasses import dataclass
from typing import Iterable, List

@dataclass
class FraudLayerParameters:
    """Parameters for a single photonic layer with an optional controlled‑phase."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]
    phase_shift: float = 0.0  # additional controlled‑phase after squeezing

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _apply_layer(modes: List, params: FraudLayerParameters, *, clip: bool) -> None:
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        Sgate(r if not clip else _clip(r, 5), phi) | modes[i]
    # controlled‑phase after squeezing
    for i in range(len(modes)):
        Rgate(params.phase_shift) | modes[i]
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        Dgate(r if not clip else _clip(r, 5), phi) | modes[i]
    for i, k in enumerate(params.kerr):
        Kgate(k if not clip else _clip(k, 1)) | modes[i]

class FraudDetectionQuantum:
    """Quantum photonic circuit for fraud detection with depth control."""
    def __init__(self, input_params: FraudLayerParameters,
                 layers: Iterable[FraudLayerParameters],
                 depth: int | None = None) -> None:
        self.input_params = input_params
        self.layers = list(layers)
        self.depth = depth if depth is not None else len(self.layers)
        self.program = self._build_program()

    def _build_program(self) -> sf.Program:
        prog = sf.Program(2)
        with prog.context as q:
            _apply_layer(q, self.input_params, clip=False)
            for i in range(self.depth):
                _apply_layer(q, self.layers[i], clip=True)
        return prog

    @property
    def trainable_params(self) -> List[float]:
        """Return a flat list of all numeric parameters for easy sweep."""
        params: List[float] = []
        for p in [self.input_params] + self.layers:
            params.extend([p.bs_theta, p.bs_phi])
            params.extend(p.phases)
            params.extend(p.squeeze_r)
            params.extend(p.squeeze_phi)
            params.extend(p.displacement_r)
            params.extend(p.displacement_phi)
            params.extend(p.kerr)
            params.append(p.phase_shift)
        return params

__all__ = ["FraudLayerParameters", "FraudDetectionQuantum"]
