"""Quantum photonic model for fraud detection with gradient support.

The module builds a Strawberry Fields program analogous to the classical
architecture and provides utilities to compute expectation values and
parameter‑shift gradients for hybrid optimisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import math
import strawberryfields as sf
from strawberryfields import Engine
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate


@dataclass
class LayerParams:
    """Parameters of a photonic layer (identical to the classical counterpart)."""
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


def _apply_layer(
    q: Sequence,
    params: LayerParams,
    *,
    clip: bool,
) -> None:
    """Insert a single photonic layer into the circuit."""
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


def _extract_params(program: sf.Program) -> list[tuple[object, str]]:
    """Return a list of (operation, attribute) tuples for all tunable parameters."""
    param_ops: list[tuple[object, str]] = []
    for op in program.ops:
        if isinstance(op, BSgate):
            param_ops.append((op, "theta"))
            param_ops.append((op, "phi"))
        elif isinstance(op, Rgate):
            param_ops.append((op, "phi"))
        elif isinstance(op, Sgate):
            param_ops.append((op, "r"))
            param_ops.append((op, "phi"))
        elif isinstance(op, Dgate):
            param_ops.append((op, "r"))
            param_ops.append((op, "phi"))
        elif isinstance(op, Kgate):
            param_ops.append((op, "k"))
    return param_ops


class FraudDetectionHybrid:
    """Quantum photonic circuit mirroring the classical fraud‑detection network.

    The circuit can be executed on a Strawberry Fields simulator, and the
    parameter‑shift rule can be used to obtain gradients for hybrid training.
    """

    def __init__(
        self,
        input_params: LayerParams,
        layers: Iterable[LayerParams],
        *,
        num_modes: int = 2,
        seed: int | None = None,
    ) -> None:
        self.input_params = input_params
        self.layers = list(layers)
        self.num_modes = num_modes
        self.seed = seed
        self.program = self._build_program()
        self.param_ops = _extract_params(self.program)

    def _build_program(self) -> sf.Program:
        prog = sf.Program(self.num_modes)
        with prog.context as q:
            _apply_layer(q, self.input_params, clip=False)
            for layer in self.layers:
                _apply_layer(q, layer, clip=True)
        return prog

    def run(self, shots: int = 1000, backend: str = "fock") -> sf.Result:
        """Execute the circuit and return a `Result` object."""
        eng = Engine(backend, backend_args=dict(dump_program=False))
        return eng.run(self.program, shots=shots)

    def expectation_photon_number(self, shots: int = 1000) -> float:
        """Return the mean photon number at the first mode."""
        res = self.run(shots)
        data = res.samples
        return float(data[:, 0].mean())

    def parameter_shift_gradients(self, shots: int = 2000) -> list[float]:
        """Compute gradients of the mean photon number w.r.t. each tunable parameter
        using the parameter‑shift rule.
        """
        grads: list[float] = []
        base_val = self.expectation_photon_number(shots)

        for op, attr in self.param_ops:
            original = getattr(op, attr)
            plus = original + math.pi / 2
            minus = original - math.pi / 2

            setattr(op, attr, plus)
            val_plus = self.expectation_photon_number(shots)

            setattr(op, attr, minus)
            val_minus = self.expectation_photon_number(shots)

            grads.append((val_plus - val_minus) / 2)

            # Reset to original value
            setattr(op, attr, original)

        return grads

__all__ = ["LayerParams", "FraudDetectionHybrid"]
