from __future__ import annotations

import numpy as np
import torch
from torch import nn
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Union

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


@dataclass
class FraudLayerParams:
    """Parameters for a fraud‑detection style fully‑connected layer."""
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


def _layer_from_params(params: FraudLayerParams, *, clip: bool) -> nn.Module:
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi], [params.squeeze_r[0], params.squeeze_r[1]]],
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
            return out * self.scale + self.shift

    return Layer()


def build_fraud_detection_model(
    input_params: FraudLayerParams,
    layers: Iterable[FraudLayerParams],
) -> nn.Sequential:
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(l, clip=True) for l in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    t = torch.as_tensor(values, dtype=torch.float32)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    return t


class FastHybridEstimator:
    """Unified estimator for both classical PyTorch models and quantum circuits."""
    def __init__(self, model: Union[nn.Module, object]) -> None:
        self.model = model
        self._is_quantum = not isinstance(model, nn.Module)

    def evaluate(
        self,
        observables: Iterable[Union[ScalarObservable, object]],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        if isinstance(self.model, nn.Module):
            return self._eval_nn(observables, parameter_sets, shots, seed)
        else:
            return self._eval_quantum(observables, parameter_sets, shots, seed)

    def _eval_nn(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        shots: int | None,
        seed: int | None,
    ) -> List[List[float]]:
        observables = list(observables) or [lambda out: out.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inp = _ensure_batch(params)
                out = self.model(inp)
                row: List[float] = []
                for obs in observables:
                    val = obs(out)
                    if isinstance(val, torch.Tensor):
                        val = float(val.mean().cpu())
                    else:
                        val = float(val)
                    row.append(val)
                results.append(row)
        return self._add_shots(results, shots, seed)

    def _eval_quantum(
        self,
        observables: Iterable[object],
        parameter_sets: Sequence[Sequence[float]],
        shots: int | None,
        seed: int | None,
    ) -> List[List[float]]:
        raw = self.model.evaluate(observables, parameter_sets)
        return self._add_shots(raw, shots, seed)

    @staticmethod
    def _add_shots(raw: List[List[float]], shots: int | None, seed: int | None) -> List[List[float]]:
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy


__all__ = ["FastHybridEstimator", "FraudLayerParams", "build_fraud_detection_model"]
