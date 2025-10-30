from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, List, Callable, Tuple

import torch
from torch import nn
import numpy as np

# --------------------------------------------------------------------------- #
# Classical photonic‑style network
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    """Parameters for a single photonic layer."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]


def _clip(value: float, bound: float) -> float:
    """Clamp a scalar to the interval [-bound, bound]."""
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """Create a single two‑node layer mirroring the photonic block."""
    weight = torch.tensor(
        [
            [params.bs_theta, params.bs_phi],
            [params.squeeze_r[0], params.squeeze_r[1]],
        ],
        dtype=torch.float32,
    )
    bias = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)

    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)

    activation = nn.Tanh()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            out = self.activation(self.linear(inputs))
            return out * self.scale + self.shift

    return Layer()


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Construct a sequential network mirroring the layered photonic model."""
    modules: List[nn.Module] = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


# --------------------------------------------------------------------------- #
# Estimation utilities
# --------------------------------------------------------------------------- #
ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


class FastEstimator:
    """Deterministic estimator with optional Gaussian shot noise."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in observables:
                    val = observable(outputs)
                    if isinstance(val, torch.Tensor):
                        val = val.mean().cpu().item()
                    row.append(float(val))
                results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [
                float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row
            ]
            noisy.append(noisy_row)
        return noisy


# --------------------------------------------------------------------------- #
# Hybrid class that exposes the classical model and estimator
# --------------------------------------------------------------------------- #
class FraudDetectionHybrid:
    """
    A fully classical fraud‑detection model built from photonic‑style layers.
    The class also provides a convenient estimator that supports deterministic
    evaluation and optional shot‑noise emulation.

    Parameters
    ----------
    input_params : FraudLayerParameters
        Parameters for the first (input) layer.
    layers : Iterable[FraudLayerParameters]
        Parameters for subsequent hidden layers.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
    ) -> None:
        self.model = build_fraud_detection_program(input_params, layers)
        self.estimator = FastEstimator(self.model)

    def evaluate(
        self,
        parameter_sets: Sequence[Sequence[float]],
        observables: Iterable[ScalarObservable] | None = None,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Evaluate the network for many input parameter sets.

        The estimator automatically adds Gaussian noise if ``shots`` is supplied;
        otherwise, it returns deterministic mean values.
        """
        if observables is None:
            observables = [lambda out: out.mean(dim=-1)]
        return self.estimator.evaluate(
            observables, parameter_sets, shots=shots, seed=seed
        )


__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "FastEstimator", "FraudDetectionHybrid"]
