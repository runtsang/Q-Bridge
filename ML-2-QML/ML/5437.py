"""Hybrid estimator combining classical neural networks with quantum‑inspired functionalities.

This module defines `HybridEstimator` that can evaluate a PyTorch model or
a hybrid classifier that contains a quantum expectation head.  It supports
batched evaluation, optional Gaussian shot noise, construction of fraud‑detection
style layers with clipping and scaling, and a flexible interface for hybrid
heads.  The API mirrors the original FastBaseEstimator while adding quantum‑
related extensions.
"""

import numpy as np
import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Callable, Optional, Union

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]
ParameterSet = Sequence[float]
ObservableSequence = Iterable[ScalarObservable]

def _ensure_batch(values: Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

@dataclass
class FraudLayerParameters:
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

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a sequential PyTorch model mirroring a fraud‑detection circuit."""
    modules: List[nn.Module] = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

class HybridEstimator:
    """Evaluate a PyTorch model for batches of parameters with optional noise.

    The estimator accepts any nn.Module.  It can be used directly for
    classical models or for hybrid models that contain a quantum expectation
    head (see `Hybrid` below).  The interface is intentionally identical to
    the original FastBaseEstimator to ease integration.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        observables: ObservableSequence,
        parameter_sets: Sequence[ParameterSet],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Compute observables for each parameter set.

        Parameters
        ----------
        observables
            Iterable of callables that map a model output tensor to a scalar.
            If empty, the mean of the output is used.
        parameter_sets
            Sequence of parameter vectors to feed to the model.
        shots
            If provided, Gaussian noise with variance 1/shots is added to
            each result to emulate measurement statistics.
        seed
            Random seed for reproducibility of shot noise.
        """
        observables = list(observables) or [lambda outputs: outputs.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for params in parameter_sets:
                inputs = _ensure_batch(params)
                outputs = self.model(inputs)
                row: List[float] = []
                for observable in observables:
                    value = observable(outputs)
                    if isinstance(value, torch.Tensor):
                        scalar = float(value.mean().cpu())
                    else:
                        scalar = float(value)
                    row.append(scalar)
                results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[float]] = []
        for row in results:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

    # ------------------------------------------------------------------
    # Convenience helpers that expose the fraud‑detection builder
    # ------------------------------------------------------------------
    @staticmethod
    def fraud_detection_model(
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        *,
        clip: bool = True,
    ) -> nn.Sequential:
        """Return a fraud‑detection model constructed from the supplied params."""
        return build_fraud_detection_program(input_params, layers)

    # ------------------------------------------------------------------
    # Compatibility helpers for hybrid quantum heads
    # ------------------------------------------------------------------
    @staticmethod
    def hybrid_head(
        in_features: int,
        shift: float = 0.0,
    ) -> nn.Module:
        """Return a simple dense head that mimics a quantum expectation layer.

        The head can be used in a hybrid network where the last layer
        performs a quantum expectation.  The shift mimics the phase shift
        applied in the QML counterpart.
        """
        class Head(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(in_features, 1)
                self.shift = shift

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
                logits = inputs.view(inputs.size(0), -1)
                return torch.sigmoid(logits + self.shift)

        return Head()
