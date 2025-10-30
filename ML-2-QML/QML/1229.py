"""Quantum photonic fraud detection circuit with readout and gradient support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import strawberryfields as sf
from strawberryfields import Engine
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate, MeasureZ
from strawberryfields.backends import BackendBase, Result


@dataclass
class FraudLayerParameters:
    """Parameters for a single photonic layer in the hybrid fraud detection circuit."""
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


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    """Construct a Strawberry Fields program for the fraud‑detection model.

    The program prepares the vacuum state, applies a sequence of photonic layers
    (beam‑splitter, squeezing, displacement, Kerr) and finally measures the photon
    number in each mode using ``MeasureZ``.  The returned program can be executed
    on any Strawberry Fields backend that supports measurement.
    """
    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
        MeasureZ | q[0]
        MeasureZ | q[1]
    return program


def evaluate(
    program: sf.Program,
    backend: BackendBase,
    shots: int = 1024,
) -> tuple[float, float]:
    """Run the program on a backend and return the mean photon numbers."""
    results: Result = backend.run(program, shots=shots)
    samples = results.samples
    mean0 = samples[:, 0].mean()
    mean1 = samples[:, 1].mean()
    return float(mean0), float(mean1)


# Simple gradient estimator using parameter‑shift rule
def parameter_shift_gradient(
    program: sf.Program,
    backend: BackendBase,
    op_index: int,
    param_name: str,
    shift: float = 0.1,
    shots: int = 1024,
) -> float:
    """Estimate the gradient of an expectation value w.r.t. a single parameter
    using the parameter‑shift rule.

    Parameters
    ----------
    program
        The program to differentiate.
    backend
        Backend to run the shifted programs.
    op_index
        Index of the operation in ``program.ops`` whose parameter is shifted.
    param_name
        Name of the parameter attribute (e.g., ``'theta'``).
    shift
        Shift size for the parameter‑shift rule.
    """
    # Clone program and shift the parameter positively
    prog_plus = sf.Program(program.num_modes)
    prog_minus = sf.Program(program.num_modes)
    for i, op in enumerate(program.ops):
        op_plus = op.copy()
        op_minus = op.copy()
        if i == op_index:
            setattr(op_plus, param_name, getattr(op, param_name) + shift)
            setattr(op_minus, param_name, getattr(op, param_name) - shift)
        prog_plus.ops.append(op_plus)
        prog_minus.ops.append(op_minus)

    exp_plus = evaluate(prog_plus, backend, shots)[0]
    exp_minus = evaluate(prog_minus, backend, shots)[0]
    return (exp_plus - exp_minus) / (2 * shift)


__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "evaluate", "parameter_shift_gradient"]
