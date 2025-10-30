from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Callable, List

import torch
from torch import nn
import numpy as np

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]


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
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()


def build_fraud_detection_model(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a sequential PyTorch model mirroring the layered structure."""
    modules: List[nn.Module] = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


class FraudDetectionModel:
    """
    Classical fraud‑detection model based on a stack of custom linear layers.
    The model is fully determined by the supplied parameters and provides
    efficient batch evaluation as well as optional shot‑noise simulation.
    """

    def __init__(self, input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters]) -> None:
        self._model = build_fraud_detection_model(input_params, layers)
        self._model.eval()

    def evaluate(
        self,
        observables: Iterable[ScalarObservable],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Return a list of results for each parameter set.

        Parameters
        ----------
        observables : iterable of callables
            Each callable receives the model output and returns a scalar
            (float or tensor). If no observables are supplied the mean
            over the last dimension is returned.
        parameter_sets : sequence of float sequences
            Each sequence is fed to the model as a batch of inputs.
        shots : int | None
            If provided the deterministic result is perturbed with Gaussian
            noise of variance 1/shots to emulate measurement sampling.
        seed : int | None
            Seed for the noise generator.
        """
        if not observables:
            observables = [lambda out: out.mean(dim=-1)]

        results: List[List[float]] = []
        rng = np.random.default_rng(seed) if shots is not None else None

        with torch.no_grad():
            for params in parameter_sets:
                inputs = torch.as_tensor(params, dtype=torch.float32)
                if inputs.ndim == 1:
                    inputs = inputs.unsqueeze(0)
                outputs = self._model(inputs)
                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    scalar = float(value.mean().cpu()) if isinstance(value, torch.Tensor) else float(value)
                    row.append(scalar)
                if shots is not None:
                    noise = rng.normal(0, 1 / np.sqrt(shots), size=len(row))
                    row = (np.array(row) + noise).tolist()
                results.append(row)
        return results

    def predict(self, inputs: Sequence[float]) -> torch.Tensor:
        """Return the raw model output for a single input."""
        self._model.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(inputs, dtype=torch.float32).unsqueeze(0)
            return self._model(tensor).squeeze(0)


__all__ = ["FraudLayerParameters", "build_fraud_detection_model", "FraudDetectionModel"]
