"""Quantum photonic fraud detection model with extended utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence, Dict, Any
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
import numpy as np

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
    clip: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if name in ("phases", "squeeze_r", "squeeze_phi",
                        "displacement_r", "displacement_phi", "kerr"):
                if not isinstance(value, (tuple, list)) or len(value)!= 2:
                    raise ValueError(f"{name} must be a 2-tuple of floats")
        if self.clip:
            self._clip_params()

    def _clip_params(self) -> None:
        bound = 5.0
        for field_name in ("bs_theta", "bs_phi", "phases", "squeeze_r",
                           "squeeze_phi", "displacement_r", "displacement_phi"):
            value = getattr(self, field_name)
            if isinstance(value, (tuple, list)):
                clipped = tuple(max(-bound, min(bound, v)) for v in value)
                setattr(self, field_name, clipped)
            else:
                setattr(self, field_name,
                        max(-bound, min(bound, value)))
        self.kerr = tuple(max(-1.0, min(1.0, k)) for k in self.kerr)

    @classmethod
    def random(cls, clip: bool = False) -> "FraudLayerParameters":
        """Generate a random parameter set for debugging or initialization."""
        rng = np.random.default_rng()
        return cls(
            bs_theta=rng.uniform(-np.pi, np.pi),
            bs_phi=rng.uniform(-np.pi, np.pi),
            phases=(rng.uniform(-np.pi, np.pi),
                    rng.uniform(-np.pi, np.pi)),
            squeeze_r=(rng.uniform(0, 2), rng.uniform(0, 2)),
            squeeze_phi=(rng.uniform(-np.pi, np.pi),
                         rng.uniform(-np.pi, np.pi)),
            displacement_r=(rng.uniform(0, 2), rng.uniform(0, 2)),
            displacement_phi=(rng.uniform(-np.pi, np.pi),
                              rng.uniform(-np.pi, np.pi)),
            kerr=(rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "bs_theta": self.bs_theta,
            "bs_phi": self.bs_phi,
            "phases": self.phases,
            "squeeze_r": self.squeeze_r,
            "squeeze_phi": self.squeeze_phi,
            "displacement_r": self.displacement_r,
            "displacement_phi": self.displacement_phi,
            "kerr": self.kerr,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FraudLayerParameters":
        """Deserialize from a dict."""
        return cls(**data)

    def to_classical_params(self) -> Dict[str, Any]:
        """Map the photonic parameters to an equivalent classical layer."""
        return {
            "bs_theta": self.bs_theta,
            "bs_phi": self.bs_phi,
            "phases": self.phases,
            "squeeze_r": self.squeeze_r,
            "squeeze_phi": self.squeeze_phi,
            "displacement_r": self.displacement_r,
            "displacement_phi": self.displacement_phi,
            "kerr": self.kerr,
        }

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

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

def compute_unitary(program: sf.Program, cutoff_dim: int = 5) -> np.ndarray:
    """Return the unitary matrix of the program using the Fock backend."""
    eng = sf.Engine("fock", backend_options={"cutoff_dim": cutoff_dim})
    result = eng.run(program)
    return result.state.unitary

__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "compute_unitary"]
