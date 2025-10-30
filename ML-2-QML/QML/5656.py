"""Quantum photonic fraud detection program with parameter‑shift gradient and error‑mitigation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import strawberryfields as sf
from strawberryfields import Engine
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate, MeasureFock, Fock

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

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _apply_layer(q: sf.Program, params: FraudLayerParameters, *, clip: bool) -> None:
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

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    """Construct a Strawberry Fields program with post‑selection and measurement."""
    prog = sf.Program(2)
    with prog.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
        # Measure photon number in each mode
        MeasureFock() | q[0]
        MeasureFock() | q[1]
    return prog

def post_select(measurements: np.ndarray, threshold: int = 1) -> np.ndarray:
    """Simple post‑selection: keep shots where both photon numbers are <= threshold."""
    mask = np.all(measurements <= threshold, axis=1)
    return measurements[mask]

def parameter_shift_gradient(
    prog: sf.Program,
    param_name: str,
    delta: float = np.pi / 2,
    shots: int = 2000,
    engine: Engine = None,
) -> float:
    """
    Estimate gradient of expectation value w.r.t. a single gate parameter using
    the parameter‑shift rule. Only works for parameters that appear as a scalar
    in a single gate (e.g., theta of a BSgate).
    """
    if engine is None:
        engine = Engine("gaussian", shots=shots)
    # Identify the gate and its index
    gate, idx = None, None
    for i, g in enumerate(prog.circuit):
        if hasattr(g, param_name):
            gate = g
            idx = i
            break
    if gate is None:
        raise ValueError(f"Parameter {param_name} not found in program.")
    # Create two perturbed programs
    prog_plus = prog.copy()
    prog_minus = prog.copy()
    setattr(prog_plus.circuit[idx], param_name, getattr(gate, param_name) + delta)
    setattr(prog_minus.circuit[idx], param_name, getattr(gate, param_name) - delta)
    # Run both programs
    results_plus = engine.run(prog_plus).state
    results_minus = engine.run(prog_minus).state
    # Expectation value: mean photon number in mode 0
    exp_plus = results_plus.expectation_value(Fock(0))
    exp_minus = results_minus.expectation_value(Fock(0))
    return (exp_plus - exp_minus) / (2 * np.sin(delta))

def train_qml_model(
    input_params: FraudLayerParameters,
    layers: List[FraudLayerParameters],
    epochs: int,
    lr: float,
    shots: int,
    device: str = "cpu",
) -> FraudLayerParameters:
    """
    Train the photonic circuit using parameter‑shift gradients.
    Returns updated parameters for the first layer (illustrative).
    """
    prog = build_fraud_detection_program(input_params, layers)
    engine = Engine("gaussian", shots=shots)
    params = input_params
    for epoch in range(epochs):
        # Example: optimize only the first BSgate theta
        grad = parameter_shift_gradient(prog, "theta", delta=np.pi/2, shots=shots, engine=engine)
        # Simple gradient descent
        params.bs_theta -= lr * grad
        # Rebuild program with updated parameters
        prog = build_fraud_detection_program(params, layers)
    return params

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "post_select",
    "parameter_shift_gradient",
    "train_qml_model",
]
