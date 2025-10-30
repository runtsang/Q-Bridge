"""FraudDetectionHybridModel – quantum branch of the hybrid fraud‑detection system.

The quantum side implements a photonic‑style variational circuit that
* reproduces the classical layer logic via a Strawberry‑Fields program,
* can be evaluated on a simulator or a real backend,
* returns the statevector for analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, Optional

import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Rgate, Sgate, Kgate

# --------------------------------------------------------------------------- #
# 1. Parameter schema – identical to the classic seed
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer (kept compatible)."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

# --------------------------------------------------------------------------- #
# 2. Helper – clip
# --------------------------------------------------------------------------- #
def _clip(value: float, bound: float) -> float:
    """Clip a parameter to a symmetric bound."""
    return max(-bound, min(bound, value))

# --------------------------------------------------------------------------- #
# 3. Layer application – same logic as the seed
# --------------------------------------------------------------------------- #
def _apply_layer(modes: Sequence, params: FraudLayerParameters, *, clip: bool) -> None:
    """Apply a single photonic layer to the modes."""
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

# --------------------------------------------------------------------------- #
# 4. Program builder
# --------------------------------------------------------------------------- #
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

# --------------------------------------------------------------------------- #
# 5. Convenience run helper
# --------------------------------------------------------------------------- #
def run_fraud_detection(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    backend: str = "gaussian",
    shots: int = 1000,
) -> sf.Result:
    """
    Compile and run the photonic fraud detection circuit.

    Parameters
    ----------
    input_params
        Parameters for the first (input) layer.
    layers
        Iterable of FraudLayerParameters for the hidden layers.
    backend
        Backend name supported by Strawberry Fields (e.g. 'gaussian',
        'fock', 'tf', 'braket').
    shots
        Number of shots for a sampling backend; ignored for state‑vector
        backends.

    Returns
    -------
    sf.Result
        The result object from the engine run, containing statevector,
        expectation values, and measurement statistics.
    """
    program = build_fraud_detection_program(input_params, layers)
    eng = sf.Engine(backend)
    result = eng.run(program, shots=shots)
    return result

__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "run_fraud_detection"]
