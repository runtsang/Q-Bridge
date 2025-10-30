from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import strawberryfields as sf
from strawberryfields.ops import BSgate, Rgate, Sgate, Dgate, Kgate
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# 1. Shared parameter container (identical to the classical side)
# ──────────────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────────────
# 2. Quantum building blocks
# ──────────────────────────────────────────────────────────────────────────────
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

def build_quantum_fraud_program(input_params: FraudLayerParameters,
                                layers: Iterable[FraudLayerParameters]) -> sf.Program:
    """Create a StrawberryFields program that mirrors the classical fraud layers."""
    prog = sf.Program(2)
    with prog.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return prog

# ──────────────────────────────────────────────────────────────────────────────
# 3. Quantum SamplerQNN
# ──────────────────────────────────────────────────────────────────────────────
def SamplerQNN() -> sf.Program:
    """A minimal two‑mode variational sampler used as a QNN."""
    prog = sf.Program(2)
    with prog.context as q:
        # Input encoding (fixed for illustration)
        Rgate(0.1) | q[0]
        Rgate(0.2) | q[1]
        # Entangling block
        BSgate(np.pi / 4, 0) | (q[0], q[1])
        # Variational layer
        Sgate(0.3) | q[0]
        Sgate(0.4) | q[1]
    return prog

__all__ = [
    "FraudLayerParameters",
    "build_quantum_fraud_program",
    "SamplerQNN",
]
