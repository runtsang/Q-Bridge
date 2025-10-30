from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Callable, Tuple

import numpy as np
import torch
from torch import nn

ScalarObs = Callable[[torch.Tensor], torch.Tensor | float]

def _batchify(values: Sequence[float]) -> torch.Tensor:
    """Convert a 1‑D sequence into a batch‑ready 2‑D tensor."""
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

@dataclass
class FraudLayerParams:
    """Parameters describing a fully‑connected fraud‑detection layer."""
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

def _build_fraud_layer(params: FraudLayerParams, *, clip: bool) -> nn.Module:
    """Create a single fraud‑detection layer with optional clipping."""
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

    class FraudLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            out = self.activation(self.linear(x))
            return out * self.scale + self.shift

    return FraudLayer()

def build_fraud_detection_model(
    input_params: FraudLayerParams,
    layers: Iterable[FraudLayerParams],
) -> nn.Sequential:
    """Construct a sequential fraud‑detection network."""
    modules: List[nn.Module] = [_build_fraud_layer(input_params, clip=False)]
    modules += [_build_fraud_layer(layer, clip=True) for layer in layers]
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

class HybridEstimator:
    """Hybrid estimator capable of evaluating arbitrary PyTorch models
    and providing optional Gaussian shot noise.
    """
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.model.eval()

    def evaluate(
        self,
        observables: Iterable[ScalarObs],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[float]]:
        """Compute deterministic outputs for every parameter set."""
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        with torch.no_grad():
            for params in parameter_sets:
                batch = _batchify(params)
                output = self.model(batch)
                row: List[float] = []
                for obs in observables:
                    val = obs(output)
                    if isinstance(val, torch.Tensor):
                        val = float(val.mean().cpu())
                    else:
                        val = float(val)
                    row.append(val)
                results.append(row)
        return results

    def evaluate_with_shots(
        self,
        observables: Iterable[ScalarObs],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Wrap evaluate with Gaussian noise to emulate finite shots."""
        raw = self.evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

__all__ = ["HybridEstimator", "FraudLayerParams", "build_fraud_detection_model"]
