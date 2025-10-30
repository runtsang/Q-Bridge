"""Fraud detection model – quantum implementation using Strawberry Fields."""

from __future__ import annotations

import typing
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
import numpy as np

@dataclass
class FraudLayerParameters:
    """Parameters of a single photonic layer (mirrors the classical seed)."""
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

class FraudDetectionModel:
    """
    Quantum analogue of the fraud‑detection architecture.

    The model builds a two‑mode Strawberry Fields program that
    applies a sequence of beamsplitter, rotation, squeezing,
    displacement, and Kerr gates.  Parameters are split into an
    input layer (unclipped) and a list of hidden layers
    (clipped to keep them physically realistic).

    Methods
    -------
    build_program
        Construct the SF program.
    simulate
        Run the program on a chosen backend and return measurement
        results or expectation values.
    param_shift_gradient
        Compute gradients of a scalar cost w.r.t. the parameters
        using the parameter‑shift rule.
    """
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: typing.Iterable[FraudLayerParameters],
        backend: str = "aer",
        shots: int = 1024,
    ) -> None:
        self.input_params = input_params
        self.layers = list(layers)
        self.backend = backend
        self.shots = shots
        self.program = self.build_program()

    def _apply_layer(self, q: sf.Program, params: FraudLayerParameters, *, clip: bool) -> None:
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

    def build_program(self) -> sf.Program:
        prog = sf.Program(2)
        with prog.context as q:
            self._apply_layer(q, self.input_params, clip=False)
            for layer in self.layers:
                self._apply_layer(q, layer, clip=True)
        return prog

    def simulate(self, meas: str = "m") -> np.ndarray:
        """Run the program on the chosen backend and return raw results."""
        eng = sf.Engine(self.backend)
        results = eng.run(self.program, shots=self.shots)
        if meas == "m":
            return np.array(results.samples)
        elif meas == "p":
            return np.array(results.states[0].probabilities())
        else:
            raise ValueError(f"Unsupported measurement type: {meas}")

    def param_shift_gradient(self, cost_fn: typing.Callable[[np.ndarray], float]) -> dict:
        """
        Compute the gradient of a scalar cost w.r.t. all parameters
        using the parameter‑shift rule.

        Parameters
        ----------
        cost_fn
            Function that accepts a single sample array and returns a scalar
            cost.  Typical use: cost_fn(samples) -> loss.

        Returns
        -------
        grads : dict
            Mapping from parameter names to gradient arrays of the same shape.
        """
        shift = np.pi / 2
        grads = {}
        # Helper to evaluate cost with a modified program
        def _cost_with_mod(modify_fn) -> float:
            prog_mod = self.build_program()
            modify_fn(prog_mod)
            eng = sf.Engine(self.backend)
            results = eng.run(prog_mod, shots=self.shots)
            return cost_fn(np.array(results.samples))

        # Iterate over all parameters of all layers
        for param_set, is_input in [(self.input_params, True)] + [(l, False) for l in self.layers]:
            for field in param_set.__dataclass_fields__:
                base = getattr(param_set, field)
                if isinstance(base, tuple):
                    shape = np.array(base).shape
                    grad = np.zeros(shape)
                    for idx in np.ndindex(shape):
                        idx_tuple = tuple(idx)
                        def shift_plus(prog):
                            val = getattr(param_set, field)
                            val = list(val)
                            val[idx] = val[idx] + shift
                            setattr(param_set, field, tuple(val))
                            # rebuild program inside the context
                            self._apply_layer(prog.context, param_set, clip=not is_input)
                        def shift_minus(prog):
                            val = getattr(param_set, field)
                            val = list(val)
                            val[idx] = val[idx] - shift
                            setattr(param_set, field, tuple(val))
                            self._apply_layer(prog.context, param_set, clip=not is_input)

                        cost_plus = _cost_with_mod(shift_plus)
                        cost_minus = _cost_with_mod(shift_minus)
                        grad[idx] = 0.5 * (cost_plus - cost_minus)
                    grads[f"{field}_{'input' if is_input else 'hidden'}"] = grad
        return grads

__all__ = ["FraudLayerParameters", "FraudDetectionModel", "_clip"]
