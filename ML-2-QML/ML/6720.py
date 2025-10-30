from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, List, Callable, Union

import numpy as np
import torch
from torch import nn

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence to a 2‑D batch tensor."""
    t = torch.as_tensor(values, dtype=torch.float32)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    return t

def _clip(value: float, bound: float) -> float:
    """Clamping helper used by the fraud‑detection layers."""
    return max(-bound, min(bound, value))

@dataclass
class FraudLayerParameters:
    """Parameters describing a fully connected layer in the classical model."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """Create a single linear‑tanh‑scaled block from parameters."""
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi],
         [params.squeeze_r[0], params.squeeze_r[1]]],
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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            out = self.activation(self.linear(inputs))
            out = out * self.scale + self.shift
            return out

    return Layer()

def build_fraud_detection_nn(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Construct a sequential PyTorch model mirroring the fraud‑detection architecture."""
    modules: List[nn.Module] = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

class FastBaseEstimator:
    """Hybrid estimator for classical neural networks with optional shot noise."""

    def __init__(
        self,
        model: nn.Module | None = None,
        *,
        input_params: FraudLayerParameters | None = None,
        layers: Iterable[FraudLayerParameters] | None = None,
    ) -> None:
        if model is None:
            if input_params is None:
                raise ValueError("Either a model or input_params must be supplied.")
            self.model = build_fraud_detection_nn(input_params, layers or [])
        else:
            self.model = model

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Compute mean observables for each parameter set, optionally adding Gaussian shot noise."""
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for obs in observables:
                    val = obs(outputs)
                    if isinstance(val, torch.Tensor):
                        scalar = float(val.mean().cpu())
                    else:
                        scalar = float(val)
                    row.append(scalar)
                results.append(row)
        if shots is None:
            return results
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noise_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noise_row)
        return noisy

__all__ = ["FastBaseEstimator", "FraudLayerParameters", "build_fraud_detection_nn"]
