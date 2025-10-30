from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Callable, Any

import torch
from torch import nn
import numpy as np


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


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
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

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            out = self.activation(self.linear(x))
            out = out * self.scale + self.shift
            return out

    return Layer()


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a sequential PyTorch model mirroring the layered structure."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


class FraudDetectionHybrid(nn.Module):
    """Hybrid fraud‑detection model that exposes a classical neural‑network backend
    and an evaluation API inspired by FastBaseEstimator."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layer_params: Sequence[FraudLayerParameters],
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.model = build_fraud_detection_program(input_params, layer_params).to(device)
        self.device = device

    def run(self, params: Sequence[float]) -> torch.Tensor:
        """Forward pass for a single batch of two‑dimensional inputs."""
        batch = torch.as_tensor(params, dtype=torch.float32, device=self.device)
        if batch.ndim == 1:
            batch = batch.unsqueeze(0)
        return self.model(batch)

    def evaluate(
        self,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor | float]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Compute observables for all parameter sets, optionally adding shot noise."""
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        rng = np.random.default_rng(seed) if shots is not None else None
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                output = self.run(params)
                row: List[float] = []
                for obs in observables:
                    val = obs(output)
                    scalar = float(val.mean().cpu()) if isinstance(val, torch.Tensor) else float(val)
                    if shots is not None:
                        scalar = float(rng.normal(scalar, max(1e-6, 1 / shots)))
                    row.append(scalar)
                results.append(row)
        return results


class FCL:
    """A lightweight fully‑connected layer used for quick sanity checks."""
    def __init__(self, n_features: int = 1) -> None:
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        x = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        out = torch.tanh(self.linear(x)).mean(dim=0)
        return out.detach().cpu().numpy()


__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "FraudDetectionHybrid",
    "FCL",
]
