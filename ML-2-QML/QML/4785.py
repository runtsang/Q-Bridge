import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
from dataclasses import dataclass
import numpy as np

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

def _apply_layer(modes, params: FraudLayerParameters, clip: bool) -> None:
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

def build_fraud_detection_quantum_program(
    input_params: FraudLayerParameters,
    layers: list[FraudLayerParameters],
) -> sf.Program:
    """
    Construct a Strawberry Fields program that mirrors the classical
    fraud‑detection pipeline.  The first layer is un‑clipped, and
    subsequent layers are clipped to keep parameters within a physical
    regime.  The program outputs a 2‑mode Gaussian state ready for
    measurement or further processing.
    """
    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program

def run_quantum_detection(program: sf.Program, shots: int = 1024):
    """
    Execute the photonic program on the Gaussian simulator and return
    measurement statistics.  The simulator returns mean photon numbers;
    for a real device, one would replace this with a measurement routine.
    """
    eng = sf.Engine("gaussian")
    state = eng.run(program).state
    return state.means  # (2,) mean photon numbers for the two modes

__all__ = ["FraudLayerParameters", "build_fraud_detection_quantum_program", "run_quantum_detection"]
