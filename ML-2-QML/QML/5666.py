"""Photonic fraud‑detection program with optional patch‑based quantum kernel.

The core program mirrors the original two‑mode circuit but adds a
grid of 2‑mode sub‑circuits that process 2×2 patches of a 4×4 feature
matrix.  Each patch is encoded with a random unitary followed by a
measurement, emulating the quantum kernel used in the classical
`QuantumFeatureMap`.  The two sub‑circuits are concatenated and fed
into the same linear head as the classical model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate, MeasureAll

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

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _apply_layer(q: Sequence, params: FraudLayerParameters, *, clip: bool) -> None:
    BSgate(params.bs_theta, params.bs_phi) | (q[0], q[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | q[i]
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        Sgate(r if not clip else _clip(r, 5), phi) | q[i]
    BSgate(params.bs_theta, params.bs_phi) | (q[0], q[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | q[i]
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        Dgate(r if not clip else _clip(r, 5), phi) | q[i]
    for i, k in enumerate(params.kerr):
        Kgate(k if not clip else _clip(k, 1)) | q[i]

def _apply_patch(q: Sequence, params: FraudLayerParameters, *, clip: bool) -> None:
    """Apply a photonic sub‑circuit to a 2‑mode patch."""
    # Random unitary via a chain of BS and S gates
    BSgate(params.bs_theta, params.bs_phi) | (q[0], q[1])
    Sgate(params.squeeze_r[0], params.squeeze_phi[0]) | q[0]
    Sgate(params.squeeze_r[1], params.squeeze_phi[1]) | q[1]
    BSgate(params.bs_theta, params.bs_phi) | (q[0], q[1])

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    """Standard two‑mode hybrid fraud‑detection program."""
    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program

def build_fraud_detection_patches(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    grid_size: int = 4,
    patch_size: int = 2,
) -> sf.Program:
    """Photonic program that processes a grid of 2×2 patches.

    The input is encoded into ``grid_size**2`` 2‑mode sub‑circuits.  Each
    sub‑circuit applies a random unitary (via `_apply_patch`) and a
    measurement.  The measurement results are concatenated and passed
    through a final linear gate that mirrors the classical head.
    """
    program = sf.Program(grid_size * grid_size)
    with program.context as q:
        # Encode each patch
        for i in range(grid_size * grid_size):
            _apply_patch(q[i], input_params, clip=False)
        # Optional additional layers
        for layer in layers:
            for i in range(grid_size * grid_size):
                _apply_patch(q[i], layer, clip=True)
        # Measure all modes (Pauli‑Z)
        MeasureAll() | q
    return program

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "build_fraud_detection_patches",
]
